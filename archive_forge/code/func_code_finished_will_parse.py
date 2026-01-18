import sys
from codeop import CommandCompiler
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from pygments.token import Generic, Token, Keyword, Name, Comment, String
from pygments.token import Error, Literal, Number, Operator, Punctuation
from pygments.token import Whitespace, _TokenType
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name
from curtsies.formatstring import FmtStr
from ..curtsiesfrontend.parse import parse
from ..repl import Interpreter as ReplInterpreter
def code_finished_will_parse(s: str, compiler: CommandCompiler) -> Tuple[bool, bool]:
    """Returns a tuple of whether the buffer could be complete and whether it
    will parse

    True, True means code block is finished and no predicted parse error
    True, False means code block is finished because a parse error is predicted
    False, True means code block is unfinished
    False, False isn't possible - an predicted error makes code block done"""
    try:
        return (bool(compiler(s)), True)
    except (ValueError, SyntaxError, OverflowError):
        return (True, False)