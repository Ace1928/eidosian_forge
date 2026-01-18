from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_child_widget(self, key) -> TreeWidget:
    """Return the widget for a given key.  Create if necessary."""
    return self.get_child_node(key).get_widget()