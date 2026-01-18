import sys
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast
import numpy as np
from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError
from PIL import __version__ as pil_version  # type: ignore
from ..core.request import URI_BYTES, InitializationError, IOMode, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike
def pillow_version() -> Tuple[int]:
    return tuple((int(x) for x in pil_version.split('.')))