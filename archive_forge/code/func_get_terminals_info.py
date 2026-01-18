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
def get_terminals_info(self, fsm_state_seq) -> Tuple[Tuple[PartialTerminalInfo, ...], Tuple[PartialTerminalInfo, ...]]:
    """Get the possible terminal symbols for an FSM state sequence."""
    terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
    final_terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
    for i, (fsm_id, fsm_reads_more, in_final) in enumerate(get_sub_fsms_from_seq(fsm_state_seq, self.fsms_to_trans_finals)):
        terminal_name = self.terminals[fsm_id].name
        info = PartialTerminalInfo(i, terminal_name, fsm_reads_more, in_final)
        terminals_and_info += (info,)
        if in_final:
            final_terminals_and_info += (info,)
    return (terminals_and_info, final_terminals_and_info)