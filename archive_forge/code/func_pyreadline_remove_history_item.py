import sys
from enum import (
from typing import (
def pyreadline_remove_history_item(pos: int) -> None:
    """
            An implementation of remove_history_item() for pyreadline3
            :param pos: The 0-based position in history to remove
            """
    saved_cursor = readline.rl.mode._history.history_cursor
    del readline.rl.mode._history.history[pos]
    if saved_cursor > pos:
        readline.rl.mode._history.history_cursor -= 1