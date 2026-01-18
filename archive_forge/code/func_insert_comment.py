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
@register('insert-comment')
def insert_comment(event):
    """
    Without numeric argument, comment all lines.
    With numeric argument, uncomment all lines.
    In any case accept the input.
    """
    buff = event.current_buffer
    if event.arg != 1:

        def change(line):
            return line[1:] if line.startswith('#') else line
    else:

        def change(line):
            return '#' + line
    buff.document = Document(text='\n'.join(map(change, buff.text.splitlines())), cursor_position=0)
    buff.accept_action.validate_and_handle(event.cli, buff)