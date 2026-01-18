from typing import Dict, Any, Optional
from ..lexer import Token, LexerThread
from ..utils import Serialize
from ..common import ParserConf, ParserCallbacks
from .lalr_analysis import LALR_Analyzer, IntParseTable, ParseTableBase
from .lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from .lalr_parser_state import ParserState, ParseConf
def parse_from_state(self, state: ParserState, last_token: Optional[Token]=None):
    """Run the main LALR parser loop

        Parameters:
            state - the initial state. Changed in-place.
            last_token - Used only for line information in case of an empty lexer.
        """
    try:
        token = last_token
        for token in state.lexer.lex(state):
            assert token is not None
            state.feed_token(token)
        end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
        return state.feed_token(end_token, True)
    except UnexpectedInput as e:
        try:
            e.interactive_parser = InteractiveParser(self, state, state.lexer)
        except NameError:
            pass
        raise e
    except Exception as e:
        if self.debug:
            print('')
            print('STATE STACK DUMP')
            print('----------------')
            for i, s in enumerate(state.state_stack):
                print('%d)' % i, s)
            print('')
        raise