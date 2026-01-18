from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
@classmethod
def from_pygments_lexer_cls(cls, lexer_cls):
    """
        Create a :class:`.RegexSync` instance for this Pygments lexer class.
        """
    patterns = {'Python': '^\\s*(class|def)\\s+', 'Python 3': '^\\s*(class|def)\\s+', 'HTML': '<[/a-zA-Z]', 'JavaScript': '\\bfunction\\b'}
    p = patterns.get(lexer_cls.name, '^')
    return cls(p)