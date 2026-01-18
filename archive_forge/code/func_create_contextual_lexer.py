from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def create_contextual_lexer(lexer_conf: LexerConf, parser, postlex, options) -> ContextualLexer:
    cls = options and options._plugins.get('ContextualLexer') or ContextualLexer
    parse_table: ParseTableBase[int] = parser._parse_table
    states: Dict[int, Collection[str]] = {idx: list(t.keys()) for idx, t in parse_table.states.items()}
    always_accept: Collection[str] = postlex.always_accept if postlex else ()
    return cls(lexer_conf, states, always_accept=always_accept)