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
def decypher_format_arg(format_name: str) -> Dict[str, str]:
    """Split format into plugin and format

    The V2 API aliases plugins and supported formats. This function
    splits these so that they can be fed separately to `iio.imopen`.

    """
    plugin = None
    extension = None
    if format_name is None:
        pass
    elif Path(format_name).suffix.lower() in known_extensions:
        extension = Path(format_name).suffix.lower()
    elif format_name in known_plugins:
        plugin = format_name
    elif format_name.upper() in known_plugins:
        plugin = format_name.upper()
    elif format_name.lower() in known_extensions:
        extension = format_name.lower()
    elif '.' + format_name.lower() in known_extensions:
        extension = '.' + format_name.lower()
    else:
        raise IndexError(f'No format known by name `{plugin}`.')
    return {'plugin': plugin, 'extension': extension}