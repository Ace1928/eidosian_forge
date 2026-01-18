from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def retina_figure(fig, base64=False, **kwargs):
    """format a figure as a pixel-doubled (retina) PNG

    If `base64` is True, return base64-encoded str instead of raw bytes
    for binary-encoded image formats

    .. versionadded:: 7.29
        base64 argument
    """
    pngdata = print_figure(fig, fmt='retina', base64=False, **kwargs)
    if pngdata is None:
        return
    w, h = _pngxy(pngdata)
    metadata = {'width': w // 2, 'height': h // 2}
    if base64:
        pngdata = b2a_base64(pngdata, newline=False).decode('ascii')
    return (pngdata, metadata)