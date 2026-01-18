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
def createStringObject(string: Union[str, bytes], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> Union[TextStringObject, ByteStringObject]:
    """Deprecated, use create_string_object."""
    deprecate_with_replacement('createStringObject', 'create_string_object', '4.0.0')
    return create_string_object(string, forced_encoding)