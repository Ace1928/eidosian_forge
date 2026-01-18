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
@staticmethod
def popup(*, rect: Union[RectangleObject, Tuple[float, float, float, float]], flags: int=0, parent: Optional[DictionaryObject]=None, open: bool=False) -> DictionaryObject:
    """
        Add a popup to the document.

        Args:
            rect:
                Specifies the clickable rectangular area as `[xLL, yLL, xUR, yUR]`
            flags:
                1 - invisible, 2 - hidden, 3 - print, 4 - no zoom,
                5 - no rotate, 6 - no view, 7 - read only, 8 - locked,
                9 - toggle no view, 10 - locked contents
            open:
                Whether the popup should be shown directly (default is False).
            parent:
                The contents of the popup. Create this via the AnnotationBuilder.

        Returns:
            A dictionary object representing the annotation.
        """
    deprecate_with_replacement('AnnotationBuilder.popup', 'pypdf.annotations.Popup', '4.0.0')
    from ..annotations import Popup
    popup = Popup(rect=rect, open=open, parent=parent)
    popup.flags = flags
    return popup