from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def next_inorder(self) -> TreeWidget | None:
    """Return the next TreeWidget depth first from this one."""
    first_child = self.first_child()
    if first_child is not None:
        return first_child
    this_node = self.get_node()
    next_node = this_node.next_sibling()
    depth = this_node.get_depth()
    while next_node is None and depth > 0:
        this_node = this_node.get_parent()
        next_node = this_node.next_sibling()
        depth -= 1
        if depth != this_node.get_depth():
            raise ValueError(depth)
    if next_node is None:
        return None
    return next_node.get_widget()