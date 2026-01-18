import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger
def read_frames(path, pix_fmt='rgb24', bpp=None, input_params=None, output_params=None, bits_per_pixel=None):
    """
    Create a generator to iterate over the frames in a video file.

    It first yields a small metadata dictionary that contains:

    * ffmpeg_version: the ffmpeg version in use (as a string).
    * codec: a hint about the codec used to encode the video, e.g. "h264".
    * source_size: the width and height of the encoded video frames.
    * size: the width and height of the frames that will be produced.
    * fps: the frames per second. Can be zero if it could not be detected.
    * duration: duration in seconds. Can be zero if it could not be detected.

    After that, it yields frames until the end of the video is reached. Each
    frame is a bytes object.

    This function makes no assumptions about the number of frames in
    the data. For one because this is hard to predict exactly, but also
    because it may depend on the provided output_params. If you want
    to know the number of frames in a video file, use count_frames_and_secs().
    It is also possible to estimate the number of frames from the fps and
    duration, but note that even if both numbers are present, the resulting
    value is not always correct.

    Example:

        gen = read_frames(path)
        meta = gen.__next__()
        for frame in gen:
            print(len(frame))

    Parameters:
        path (str): the filename of the file to read from.
        pix_fmt (str): the pixel format of the frames to be read.
            The default is "rgb24" (frames are uint8 RGB images).
        input_params (list): Additional ffmpeg input command line parameters.
        output_params (list): Additional ffmpeg output command line parameters.
        bits_per_pixel (int): The number of bits per pixel in the output frames.
            This depends on the given pix_fmt. Default is 24 (RGB)
        bpp (int): DEPRECATED, USE bits_per_pixel INSTEAD. The number of bytes per pixel in the output frames.
            This depends on the given pix_fmt. Some pixel formats like yuv420p have 12 bits per pixel
            and cannot be set in bytes as integer. For this reason the bpp argument is deprecated.
    """
    if isinstance(path, pathlib.PurePath):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError('Video path must be a string or pathlib.Path.')
    pix_fmt = pix_fmt or 'rgb24'
    bpp = bpp or 3
    bits_per_pixel = bits_per_pixel or bpp * 8
    input_params = input_params or []
    output_params = output_params or []
    assert isinstance(pix_fmt, str), 'pix_fmt must be a string'
    assert isinstance(bits_per_pixel, int), 'bpp and bits_per_pixel must be an int'
    assert isinstance(input_params, list), 'input_params must be a list'
    assert isinstance(output_params, list), 'output_params must be a list'
    pre_output_params = ['-pix_fmt', pix_fmt, '-vcodec', 'rawvideo', '-f', 'image2pipe']
    cmd = [get_ffmpeg_exe()]
    cmd += input_params + ['-i', path]
    cmd += pre_output_params + output_params + ['-']
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs(prevent_sigint=True))
    log_catcher = LogCatcher(process.stderr)
    stop_policy = 'timeout'
    try:
        etime = time.time() + 10.0
        while log_catcher.is_alive() and (not log_catcher.header) and (time.time() < etime):
            time.sleep(0.01)
        if not log_catcher.header:
            err2 = log_catcher.get_text(0.2)
            fmt = 'Could not load meta information\n=== stderr ===\n{}'
            raise IOError(fmt.format(err2))
        elif 'No such file or directory' in log_catcher.header:
            raise IOError('{} not found! Wrong path?'.format(path))
        meta = parse_ffmpeg_header(log_catcher.header)
        yield meta
        width, height = meta['size']
        framesize_bits = width * height * bits_per_pixel
        framesize_bytes = framesize_bits / 8
        assert framesize_bytes.is_integer(), 'incorrect bits_per_pixel, framesize in bytes must be an int'
        framesize_bytes = int(framesize_bytes)
        framenr = 0
        while True:
            framenr += 1
            try:
                bb = bytes()
                while len(bb) < framesize_bytes:
                    extra_bytes = process.stdout.read(framesize_bytes - len(bb))
                    if not extra_bytes:
                        if len(bb) == 0:
                            return
                        else:
                            raise RuntimeError('End of file reached before full frame could be read.')
                    bb += extra_bytes
                yield bb
            except Exception as err:
                err1 = str(err)
                err2 = log_catcher.get_text(0.4)
                fmt = 'Could not read frame {}:\n{}\n=== stderr ===\n{}'
                raise RuntimeError(fmt.format(framenr, err1, err2))
    except GeneratorExit:
        pass
    except Exception:
        raise
    except BaseException:
        stop_policy = 'kill'
        raise
    finally:
        log_catcher.stop_me()
        if process.poll() is None:
            try:
                process.stdout.close()
                process.stdin.close()
            except Exception as err:
                logger.warning('Error while attempting stop ffmpeg (r): ' + str(err))
            if stop_policy == 'timeout':
                try:
                    etime = time.time() + 1.5
                    while time.time() < etime and process.poll() is None:
                        time.sleep(0.01)
                finally:
                    if process.poll() is None:
                        logger.warning('We had to kill ffmpeg to stop it.')
                        process.kill()
            else:
                process.kill()