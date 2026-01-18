from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
def create_line_generator(start_lineno, column=0):
    """
            Create a generator that yields the lexed lines.
            Each iteration it yields a (line_number, [(token, text), ...]) tuple.
            """

    def get_tokens():
        text = '\n'.join(document.lines[start_lineno:])[column:]
        for _, t, v in self.pygments_lexer.get_tokens_unprocessed(text):
            yield (t, v)
    return enumerate(split_lines(get_tokens()), start_lineno)