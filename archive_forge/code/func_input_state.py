import codecs
import logging
import re
from typing import List, Optional, Union
from .charsetgroupprober import CharSetGroupProber
from .charsetprober import CharSetProber
from .enums import InputState, LanguageFilter, ProbingState
from .escprober import EscCharSetProber
from .latin1prober import Latin1Prober
from .macromanprober import MacRomanProber
from .mbcsgroupprober import MBCSGroupProber
from .resultdict import ResultDict
from .sbcsgroupprober import SBCSGroupProber
from .utf1632prober import UTF1632Prober
@property
def input_state(self) -> int:
    return self._input_state