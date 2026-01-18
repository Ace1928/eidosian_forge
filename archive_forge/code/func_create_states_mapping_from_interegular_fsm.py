from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
def create_states_mapping_from_interegular_fsm(fsm: interegular.fsm.FSM, cacheable_vocabulary: Tuple[Tuple[str, int], ...]) -> Tuple[dict, set]:
    """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
    regex_fsm, _ = make_deterministic_fsm(fsm.reduce())
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(regex_fsm, tokenizer)
    if not any((regex_fsm.finals.intersection(v.values()) for v in states_to_token_maps.values())):
        raise ValueError('The vocabulary does not allow us to build a sequence that matches the input regex')
    return (states_to_token_maps, empty_token_ids)