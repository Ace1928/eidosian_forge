from typing import (
import pyperclip  # type: ignore[import]
def get_paste_buffer() -> str:
    """Get the contents of the clipboard / paste buffer.

    :return: contents of the clipboard
    """
    pb_str = cast(str, pyperclip.paste())
    return pb_str