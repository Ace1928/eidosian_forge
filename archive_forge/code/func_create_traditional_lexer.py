from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .parsers.grammar_analysis import GrammarAnalyzer
from .lexer import LexerThread, TraditionalLexer, ContextualLexer, Lexer, Token, TerminalDef
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf
import re
def create_traditional_lexer(lexer_conf, parser, postlex):
    return TraditionalLexer(lexer_conf)