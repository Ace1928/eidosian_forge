from __future__ import unicode_literals
from six import text_type
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.token import Token
from .utils import token_list_len
from .processors import Processor, Transformation
@classmethod
def from_message(cls, message='> '):
    """
        Create a default prompt with a static message text.
        """
    assert isinstance(message, text_type)

    def get_message_tokens(cli):
        return [(Token.Prompt, message)]
    return cls(get_message_tokens)