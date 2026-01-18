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
@register('backward-kill-word')
def backward_kill_word(event):
    """
    Kills the word before point, using "not a letter nor a digit" as a word boundary.
    Usually bound to M-Del or M-Backspace.
    """
    unix_word_rubout(event, WORD=False)