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
def mimwrite(uri, ims, format=None, **kwargs):
    """mimwrite(uri, ims, format=None, **kwargs)

    Write multiple images to the specified file.

    Parameters
    ----------
    uri : {str, pathlib.Path, file}
        The resource to write the images to, e.g. a filename, pathlib.Path
        or file object, see the docs for more info.
    ims : sequence of numpy arrays
        The image data. Each array must be NxM, NxMx3 or NxMx4.
    format : str
        The format to use to read the file. By default imageio selects
        the appropriate for you based on the filename and its contents.
    kwargs : ...
        Further keyword arguments are passed to the writer. See :func:`.help`
        to see what arguments are available for a particular format.
    """
    if not is_batch(ims):
        raise ValueError('Image data must be a sequence of ndimages.')
    imopen_args = decypher_format_arg(format)
    imopen_args['legacy_mode'] = True
    with imopen(uri, 'wI', **imopen_args) as file:
        return file.write(ims, is_batch=True, **kwargs)