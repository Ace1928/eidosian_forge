from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_indented_widget(self) -> Padding[Text | Columns]:
    widget = self.get_inner_widget()
    if not self.is_leaf:
        widget = Columns([(1, [self.unexpanded_icon, self.expanded_icon][self.expanded]), widget], dividechars=1)
    indent_cols = self.get_indent_cols()
    return Padding(widget, width=(WHSettings.RELATIVE, 100), left=indent_cols)