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
def mimread(uri, format=None, memtest=MEMTEST_DEFAULT_MIM, **kwargs):
    """mimread(uri, format=None, memtest="256MB", **kwargs)

    Reads multiple images from the specified file. Returns a list of
    numpy arrays, each with a dict of meta data at its 'meta' attribute.

    Parameters
    ----------
    uri : {str, pathlib.Path, bytes, file}
        The resource to load the images from, e.g. a filename,pathlib.Path,
        http address or file object, see the docs for more info.
    format : str
        The format to use to read the file. By default imageio selects
        the appropriate for you based on the filename and its contents.
    memtest : {bool, int, float, str}
        If truthy, this function will raise an error if the resulting
        list of images consumes greater than the amount of memory specified.
        This is to protect the system from using so much memory that it needs
        to resort to swapping, and thereby stall the computer. E.g.
        ``mimread('hunger_games.avi')``.

        If the argument is a number, that will be used as the threshold number
        of bytes.

        If the argument is a string, it will be interpreted as a number of bytes with
        SI/IEC prefixed units (e.g. '1kB', '250MiB', '80.3YB').

        - Units are case sensitive
        - k, M etc. represent a 1000-fold change, where Ki, Mi etc. represent 1024-fold
        - The "B" is optional, but if present, must be capitalised

        If the argument is True, the default will be used, for compatibility reasons.

        Default: '256MB'
    kwargs : ...
        Further keyword arguments are passed to the reader. See :func:`.help`
        to see what arguments are available for a particular format.
    """
    nbyte_limit = to_nbytes(memtest, MEMTEST_DEFAULT_MIM)
    images = list()
    nbytes = 0
    imopen_args = decypher_format_arg(format)
    imopen_args['legacy_mode'] = True
    with imopen(uri, 'rI', **imopen_args) as file:
        for image in file.iter(**kwargs):
            images.append(image)
            nbytes += image.nbytes
            if nbytes > nbyte_limit:
                raise RuntimeError('imageio.mimread() has read over {}B of image data.\nStopped to avoid memory problems. Use imageio.get_reader(), increase threshold, or memtest=False'.format(int(nbyte_limit)))
    if len(images) == 1 and is_batch(images[0]):
        images = [*images[0]]
    return images