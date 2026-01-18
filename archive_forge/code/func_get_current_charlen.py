import logging
from .codingstatemachinedict import CodingStateMachineDict
from .enums import MachineState
def get_current_charlen(self) -> int:
    return self._curr_char_len