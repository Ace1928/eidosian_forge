import math
import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ._utils import (
from .constants import CcittFaxDecodeParameters as CCITT
from .constants import ColorSpaces
from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import LzwFilterParameters as LZW
from .constants import StreamAttributes as SA
from .errors import DeprecationError, PdfReadError, PdfStreamError
from .generic import (
def decodeStreamData(stream: Any) -> Union[str, bytes]:
    """Deprecated. Use decode_stream_data."""
    deprecate_with_replacement('decodeStreamData', 'decode_stream_data', '4.0.0')
    return decode_stream_data(stream)