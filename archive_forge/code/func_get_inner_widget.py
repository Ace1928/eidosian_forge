from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_inner_widget(self) -> Text:
    if self._innerwidget is None:
        self._innerwidget = self.load_inner_widget()
    return self._innerwidget