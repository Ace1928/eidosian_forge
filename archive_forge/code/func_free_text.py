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
def free_text(text: str, rect: Union[RectangleObject, Tuple[float, float, float, float]], font: str='Helvetica', bold: bool=False, italic: bool=False, font_size: str='14pt', font_color: str='000000', border_color: Optional[str]='000000', background_color: Optional[str]='ffffff') -> DictionaryObject:
    """
        Add text in a rectangle to a page.

        Args:
            text: Text to be added
            rect: array of four integers ``[xLL, yLL, xUR, yUR]``
                specifying the clickable rectangular area
            font: Name of the Font, e.g. 'Helvetica'
            bold: Print the text in bold
            italic: Print the text in italic
            font_size: How big the text will be, e.g. '14pt'
            font_color: Hex-string for the color, e.g. cdcdcd
            border_color: Hex-string for the border color, e.g. cdcdcd.
                Use ``None`` for no border.
            background_color: Hex-string for the background of the annotation,
                e.g. cdcdcd. Use ``None`` for transparent background.

        Returns:
            A dictionary object representing the annotation.
        """
    deprecate_with_replacement('AnnotationBuilder.free_text', 'pypdf.annotations.FreeText', '4.0.0')
    from ..annotations import FreeText
    return FreeText(text=text, rect=rect, font=font, bold=bold, italic=italic, font_size=font_size, font_color=font_color, background_color=background_color, border_color=border_color)