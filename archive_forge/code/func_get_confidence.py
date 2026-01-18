from typing import Optional, Union
from .charsetprober import CharSetProber
from .codingstatemachine import CodingStateMachine
from .enums import LanguageFilter, MachineState, ProbingState
from .escsm import (
def get_confidence(self) -> float:
    return 0.99 if self._detected_charset else 0.0