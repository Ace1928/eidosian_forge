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
@register('prefix-meta')
def prefix_meta(event):
    """
    Metafy the next character typed. This is for keyboards without a meta key.

    Sometimes people also want to bind other keys to Meta, e.g. 'jj'::

        registry.add_key_binding('j', 'j', filter=ViInsertMode())(prefix_meta)
    """
    event.cli.input_processor.feed(KeyPress(Keys.Escape))