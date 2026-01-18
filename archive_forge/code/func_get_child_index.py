from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def get_child_index(self, key: Hashable) -> int:
    try:
        return self.get_child_keys().index(key)
    except ValueError as exc:
        raise TreeWidgetError(f"Can't find key {key} in ParentNode {self.get_key()}\nParentNode items: {self.get_child_keys()!s}").with_traceback(exc.__traceback__) from exc