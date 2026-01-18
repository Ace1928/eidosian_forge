from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def load_child_node(self, key: Hashable) -> TreeNode:
    """Load the child node for a given key (virtual function)"""
    raise TreeWidgetError('virtual function.  Implement in subclass')