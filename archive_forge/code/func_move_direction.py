import builtins
import sys
from ...utils.imports import _is_package_available
from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP
def move_direction(self, direction: Direction, num_spaces: int=1):
    """Should not be directly called, used to move a direction of either up or down"""
    old_position = self.position
    if direction == Direction.DOWN:
        if self.position + 1 >= len(self.choices):
            return
        self.position += num_spaces
    else:
        if self.position - 1 < 0:
            return
        self.position -= num_spaces
    clear_line()
    self.print_choice(old_position)
    move_cursor(num_spaces, direction.name)
    self.print_choice(self.position)