import re
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict
import numpy as np
from imageio.core.legacy_plugin_wrapper import LegacyPlugin
from imageio.core.util import Array
from imageio.core.v3_plugin_api import PluginV3
from . import formats
from .config import known_extensions, known_plugins
from .core import RETURN_BYTES
from .core.imopen import imopen
def imwrite(uri, im, format=None, **kwargs):
    """imwrite(uri, im, format=None, **kwargs)

    Write an image to the specified file.

    Parameters
    ----------
    uri : {str, pathlib.Path, file}
        The resource to write the image to, e.g. a filename, pathlib.Path
        or file object, see the docs for more info.
    im : numpy.ndarray
        The image data. Must be NxM, NxMx3 or NxMx4.
    format : str
        The format to use to write the file. By default imageio selects
        the appropriate for you based on the filename and its contents.
    kwargs : ...
        Further keyword arguments are passed to the writer. See :func:`.help`
        to see what arguments are available for a particular format.
    """
    imt = type(im)
    im = np.asarray(im)
    if not np.issubdtype(im.dtype, np.number):
        raise ValueError('Image is not numeric, but {}.'.format(imt.__name__))
    if is_batch(im) or im.ndim < 2:
        raise ValueError('Image must be 2D (grayscale, RGB, or RGBA).')
    imopen_args = decypher_format_arg(format)
    imopen_args['legacy_mode'] = True
    with imopen(uri, 'wi', **imopen_args) as file:
        return file.write(im, **kwargs)