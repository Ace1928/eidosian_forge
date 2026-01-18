import warnings
from typing import TYPE_CHECKING, List, NewType
from outlines.fsm.guide import CFGGuide, RegexGuide, StopAtEOSGuide
def allowed_token_ids(self, state: FSMState) -> List[int]:
    return self.get_next_instruction(state).tokens