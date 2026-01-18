from __future__ import unicode_literals
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.selection import PasteMode
from six.moves import range
import six
from .completion import generate_completions, display_completions_like_readline
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
@register('kill-line')
def kill_line(event):
    """
    Kill the text from the cursor to the end of the line.

    If we are at the end of the line, this should remove the newline.
    (That way, it is possible to delete multiple lines by executing this
    command multiple times.)
    """
    buff = event.current_buffer
    if event.arg < 0:
        deleted = buff.delete_before_cursor(count=-buff.document.get_start_of_line_position())
    elif buff.document.current_char == '\n':
        deleted = buff.delete(1)
    else:
        deleted = buff.delete(count=buff.document.get_end_of_line_position())
    event.cli.clipboard.set_text(deleted)