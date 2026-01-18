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
@register('backward-delete-char')
def backward_delete_char(event):
    """ Delete the character behind the cursor. """
    if event.arg < 0:
        deleted = event.current_buffer.delete(count=-event.arg)
    else:
        deleted = event.current_buffer.delete_before_cursor(count=event.arg)
    if not deleted:
        event.cli.output.bell()