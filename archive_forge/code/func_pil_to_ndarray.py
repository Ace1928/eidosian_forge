import numpy as np
from PIL import Image
from ...util import img_as_ubyte, img_as_uint
def pil_to_ndarray(image, dtype=None, img_num=None):
    """Import a PIL Image object to an ndarray, in memory.

    Parameters
    ----------
    Refer to ``imread``.

    """
    try:
        image.getdata()[0]
    except OSError as e:
        site = 'http://pillow.readthedocs.org/en/latest/installation.html#external-libraries'
        pillow_error_message = str(e)
        error_message = f"Could not load '{image.filename}' \nReason: '{pillow_error_message}'\nPlease see documentation at: {site}"
        raise ValueError(error_message)
    frames = []
    grayscale = None
    i = 0
    while 1:
        try:
            image.seek(i)
        except EOFError:
            break
        frame = image
        if img_num is not None and img_num != i:
            image.getdata()[0]
            i += 1
            continue
        if image.format == 'PNG' and image.mode == 'I' and (dtype is None):
            dtype = 'uint16'
        if image.mode == 'P':
            if grayscale is None:
                grayscale = _palette_is_grayscale(image)
            if grayscale:
                frame = image.convert('L')
            elif image.format == 'PNG' and 'transparency' in image.info:
                frame = image.convert('RGBA')
            else:
                frame = image.convert('RGB')
        elif image.mode == '1':
            frame = image.convert('L')
        elif 'A' in image.mode:
            frame = image.convert('RGBA')
        elif image.mode == 'CMYK':
            frame = image.convert('RGB')
        if image.mode.startswith('I;16'):
            shape = image.size
            dtype = '>u2' if image.mode.endswith('B') else '<u2'
            if 'S' in image.mode:
                dtype = dtype.replace('u', 'i')
            frame = np.frombuffer(frame.tobytes(), dtype)
            frame.shape = shape[::-1]
        else:
            frame = np.array(frame, dtype=dtype)
        frames.append(frame)
        i += 1
        if img_num is not None:
            break
    if hasattr(image, 'fp') and image.fp:
        image.fp.close()
    if img_num is None and len(frames) > 1:
        return np.array(frames)
    elif frames:
        return frames[0]
    elif img_num:
        raise IndexError(f'Could not find image  #{img_num}')