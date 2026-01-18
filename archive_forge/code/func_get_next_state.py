from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
def get_next_state(self, state: int, token_id: int) -> int:
    """Update the state of the guide.

        Transitions the underlying regex FSM to its next state.
        If at max tokens or EOS token, transition permanently to the final state.
        Update stored partial generations for subsequent incremental parsing.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.
        """
    if token_id == self.tokenizer.eos_token_id or state == self.final_state:
        return self.final_state
    self.generation += self.tokenizer.decode([token_id])[0]
    if self.check_last:
        if token_id in self.proposal_last:
            return self.regex_fsm_last.get_next_state(state, token_id)
        self.check_last = False
    if self.reset_state:
        self.reset_state = False
        state = self.start_state
    return self.regex_fsm.get_next_state(state, token_id)