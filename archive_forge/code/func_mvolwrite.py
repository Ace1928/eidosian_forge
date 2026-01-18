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
def mvolwrite(uri, ims, format=None, **kwargs):
    """mvolwrite(uri, vols, format=None, **kwargs)

    Write multiple volumes to the specified file.

    Parameters
    ----------
    uri : {str, pathlib.Path, file}
        The resource to write the volumes to, e.g. a filename, pathlib.Path
        or file object, see the docs for more info.
    ims : sequence of numpy arrays
        The image data. Each array must be NxMxL (or NxMxLxK if each
        voxel is a tuple).
    format : str
        The format to use to read the file. By default imageio selects
        the appropriate for you based on the filename and its contents.
    kwargs : ...
        Further keyword arguments are passed to the writer. See :func:`.help`
        to see what arguments are available for a particular format.
    """
    for im in ims:
        if not is_volume(im):
            raise ValueError('Image must be 3D, or 4D if each voxel is a tuple.')
    imopen_args = decypher_format_arg(format)
    imopen_args['legacy_mode'] = True
    with imopen(uri, 'wV', **imopen_args) as file:
        return file.write(ims, is_batch=True, **kwargs)