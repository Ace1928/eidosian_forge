from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_prev(self, start_from) -> tuple[TreeWidget, TreeNode] | tuple[None, None]:
    target = start_from.get_widget().prev_inorder()
    if target is None:
        return (None, None)
    return (target, target.get_node())