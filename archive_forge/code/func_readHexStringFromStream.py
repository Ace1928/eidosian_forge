from typing import Dict, List, Optional, Tuple, Union
from .._utils import StreamType, deprecate_with_replacement
from ..constants import OutlineFontFlag
from ._base import (
from ._data_structures import (
from ._fit import Fit
from ._outline import OutlineItem
from ._rectangle import RectangleObject
from ._utils import (
from ._viewerpref import ViewerPreferences
def readHexStringFromStream(stream: StreamType) -> Union['TextStringObject', 'ByteStringObject']:
    """Deprecated, use read_hex_string_from_stream."""
    deprecate_with_replacement('readHexStringFromStream', 'read_hex_string_from_stream', '4.0.0')
    return read_hex_string_from_stream(stream)