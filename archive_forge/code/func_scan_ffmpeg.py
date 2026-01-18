from .utils import check_output, where
import os
import warnings
import numpy as np
def scan_ffmpeg():
    global _FFMPEG_MAJOR_VERSION
    global _FFMPEG_MINOR_VERSION
    global _FFMPEG_PATCH_VERSION
    global _FFMPEG_SUPPORTED_DECODERS
    global _FFMPEG_SUPPORTED_ENCODERS
    _FFMPEG_MAJOR_VERSION = '0'
    _FFMPEG_MINOR_VERSION = '0'
    _FFMPEG_PATCH_VERSION = '0'
    _FFMPEG_SUPPORTED_DECODERS = []
    _FFMPEG_SUPPORTED_ENCODERS = []
    try:
        version = check_output([os.path.join(_FFMPEG_PATH, _FFMPEG_APPLICATION), '-version'])
        firstline = version.split(b'\n')[0]
        version = firstline.split(b' ')[2].strip()
        versionparts = version.split(b'.')
        if version[0] == b'N':
            _FFMPEG_MAJOR_VERSION = version
        else:
            _FFMPEG_MAJOR_VERSION = versionparts[0]
            _FFMPEG_MINOR_VERSION = versionparts[1]
            if len(versionparts) > 2:
                _FFMPEG_PATCH_VERSION = versionparts[2]
    except:
        pass
    _FFMPEG_SUPPORTED_DECODERS = [b'.264', b'.265', b'.302', b'.3g2', b'.3gp', b'.722', b'.aa', b'.aa3', b'.aac', b'.ac3', b'.acm', b'.adf', b'.adp', b'.ads', b'.adx', b'.aea', b'.afc', b'.aif', b'.aifc', b'.aiff', b'.al', b'.amr', b'.ans', b'.ape', b'.apl', b'.apng', b'.aqt', b'.art', b'.asc', b'.asf', b'.ass', b'.ast', b'.au', b'.avc', b'.avi', b'.avr', b'.bcstm', b'.bfstm', b'.bin', b'.bit', b'.bmp', b'.bmv', b'.brstm', b'.caf', b'.cavs', b'.cdata', b'.cdg', b'.cdxl', b'.cgi', b'.cif', b'.daud', b'.dif', b'.diz', b'.dnxhd', b'.dpx', b'.drc', b'.dss', b'.dtk', b'.dts', b'.dtshd', b'.dv', b'.eac3', b'.fap', b'.ffm', b'.ffmeta', b'.flac', b'.flm', b'.flv', b'.fsb', b'.g722', b'.g723_1', b'.g729', b'.genh', b'.gif', b'.gsm', b'.gxf', b'.h261', b'.h263', b'.h264', b'.h265', b'.h26l', b'.hevc', b'.ice', b'.ico', b'.idf', b'.idx', b'.im1', b'.im24', b'.im8', b'.ircam', b'.ivf', b'.ivr', b'.j2c', b'.j2k', b'.jls', b'.jp2', b'.jpeg', b'.jpg', b'.js', b'.jss', b'.lbc', b'.ljpg', b'.lrc', b'.lvf', b'.m2a', b'.m2t', b'.m2ts', b'.m3u8', b'.m4a', b'.m4v', b'.mac', b'.mj2', b'.mjpeg', b'.mjpg', b'.mk3d', b'.mka', b'.mks', b'.mkv', b'.mlp', b'.mmf', b'.mov', b'.mp2', b'.mp3', b'.mp4', b'.mpa', b'.mpc', b'.mpeg', b'.mpg', b'.mpl2', b'.mpo', b'.msf', b'.mts', b'.mvi', b'.mxf', b'.mxg', b'.nfo', b'.nist', b'.nut', b'.ogg', b'.ogv', b'.oma', b'.omg', b'.paf', b'.pam', b'.pbm', b'.pcx', b'.pgm', b'.pgmyuv', b'.pix', b'.pjs', b'.png', b'.ppm', b'.pvf', b'.qcif', b'.ra', b'.ras', b'.rco', b'.rcv', b'.rgb', b'.rm', b'.roq', b'.rs', b'.rsd', b'.rso', b'.rt', b'.sami', b'.sb', b'.sbg', b'.sdr2', b'.sf', b'.sgi', b'.shn', b'.sln', b'.smi', b'.son', b'.sox', b'.spdif', b'.sph', b'.srt', b'.ss2', b'.ssa', b'.stl', b'.str', b'.sub', b'.sun', b'.sunras', b'.sup', b'.svag', b'.sw', b'.swf', b'.tak', b'.tco', b'.tga', b'.thd', b'.tif', b'.tiff', b'.ts', b'.tta', b'.txt', b'.ub', b'.ul', b'.uw', b'.v', b'.v210', b'.vag', b'.vb', b'.vc1', b'.viv', b'.voc', b'.vpk', b'.vqe', b'.vqf', b'.vql', b'.vt', b'.vtt', b'.w64', b'.wav', b'.webm', b'.wma', b'.wmv', b'.wtv', b'.wv', b'.xbm', b'.xface', b'.xl', b'.xml', b'.xvag', b'.xwd', b'.y', b'.y4m', b'.yop', b'.yuv', b'.yuv10', b'.raw', b'.iso']
    _FFMPEG_SUPPORTED_ENCODERS = [b'., A64', b'.264', b'.265', b'.302', b'.3g2', b'.3gp', b'.722', b'.a64', b'.aa3', b'.aac', b'.ac3', b'.adts', b'.adx', b'.afc', b'.aif', b'.aifc', b'.aiff', b'.al', b'.amr', b'.apng', b'.asf', b'.ass', b'.ast', b'.au', b'.avc', b'.avi', b'.bit', b'.bmp', b'.caf', b'.cavs', b'.chk', b'.cif', b'.daud', b'.dif', b'.dnxhd', b'.dpx', b'.drc', b'.dts', b'.dv', b'.dvd', b'.eac3', b'.f4v', b'.ffm', b'.ffmeta', b'.flac', b'.flm', b'.flv', b'.g722', b'.g723_1', b'.gif', b'.gxf', b'.h261', b'.h263', b'.h264', b'.h265', b'.h26l', b'.hevc', b'.ico', b'.im1', b'.im24', b'.im8', b'.ircam', b'.isma', b'.ismv', b'.ivf', b'.j2c', b'.j2k', b'.jls', b'.jp2', b'.jpeg', b'.jpg', b'.js', b'.jss', b'.latm', b'.lbc', b'.ljpg', b'.loas', b'.lrc', b'.m1v', b'.m2a', b'.m2t', b'.m2ts', b'.m2v', b'.m3u8', b'.m4a', b'.m4v', b'.mj2', b'.mjpeg', b'.mjpg', b'.mk3d', b'.mka', b'.mks', b'.mkv', b'.mlp', b'.mmf', b'.mov', b'.mp2', b'.mp3', b'.mp4', b'.mpa', b'.mpeg', b'.mpg', b'.mpo', b'.mts', b'.mxf', b'.nut', b'.oga', b'.ogg', b'.ogv', b'.oma', b'.omg', b'.opus', b'.pam', b'.pbm', b'.pcx', b'.pgm', b'.pgmyuv', b'.pix', b'.png', b'.ppm', b'.psp', b'.qcif', b'.ra', b'.ras', b'.rco', b'.rcv', b'.rgb', b'.rm', b'.roq', b'.rs', b'.rso', b'.sb', b'.sf', b'.sgi', b'.sox', b'.spdif', b'.spx', b'.srt', b'.ssa', b'.sub', b'.sun', b'.sunras', b'.sw', b'.swf', b'.tco', b'.tga', b'.thd', b'.tif', b'.tiff', b'.ts', b'.ub', b'.ul', b'.uw', b'.vc1', b'.vob', b'.voc', b'.vtt', b'.w64', b'.wav', b'.webm', b'.webp', b'.wma', b'.wmv', b'.wtv', b'.wv', b'.xbm', b'.xface', b'.xml', b'.xwd', b'.y', b'.y4m', b'.yuv', b'.raw']