from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterator, Optional, Set, Tuple, Union
import interegular
from interegular.fsm import FSM
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import LexError, UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import (
from lark.parser_frontends import (
from lark.parsers.lalr_analysis import (
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from outlines.fsm.regex import (
def feed_token_no_stack(self, token, is_end=False):
    """
        This is a copy of `ParserState.feed_token` with all the value stack
        steps removed.  Since we're not exactly parsing in order to obtain a
        CST or anything similar, we can avoid the growing expense of tracking
        the parse tree.
        """
    state_stack = self.state_stack
    states = self.parse_conf.states
    end_state = self.parse_conf.end_state
    while True:
        state = state_stack[-1]
        try:
            action, arg = states[state][token.type]
        except KeyError:
            expected = {s for s in states[state].keys() if s.isupper()}
            raise UnexpectedToken(token, expected, state=self, interactive_parser=None)
        assert arg != end_state
        if action is Shift:
            assert not is_end
            state_stack.append(arg)
            return
        else:
            rule = arg
            size = len(rule.expansion)
            if size:
                del state_stack[-size:]
            _action, new_state = states[state_stack[-1]][rule.origin.name]
            assert _action is Shift
            state_stack.append(new_state)
            if is_end and state_stack[-1] == end_state:
                return