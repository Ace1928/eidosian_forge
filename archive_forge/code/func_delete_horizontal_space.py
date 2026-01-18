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
@register('delete-horizontal-space')
def delete_horizontal_space(event):
    """ Delete all spaces and tabs around point. """
    buff = event.current_buffer
    text_before_cursor = buff.document.text_before_cursor
    text_after_cursor = buff.document.text_after_cursor
    delete_before = len(text_before_cursor) - len(text_before_cursor.rstrip('\t '))
    delete_after = len(text_after_cursor) - len(text_after_cursor.lstrip('\t '))
    buff.delete_before_cursor(count=delete_before)
    buff.delete(count=delete_after)