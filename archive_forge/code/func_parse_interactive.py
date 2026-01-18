from typing import Dict, Any, Optional
from ..lexer import Token, LexerThread
from ..utils import Serialize
from ..common import ParserConf, ParserCallbacks
from .lalr_analysis import LALR_Analyzer, IntParseTable, ParseTableBase
from .lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from .lalr_parser_state import ParserState, ParseConf
def parse_interactive(self, lexer: LexerThread, start: str):
    return self.parser.parse(lexer, start, start_interactive=True)