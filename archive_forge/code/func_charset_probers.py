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
def charset_probers(self) -> List[CharSetProber]:
    return self._charset_probers