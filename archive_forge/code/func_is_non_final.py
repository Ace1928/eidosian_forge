from typing import Optional, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
from .sbcharsetprober import SingleByteCharSetProber
def is_non_final(self, c: int) -> bool:
    return c in [self.NORMAL_KAF, self.NORMAL_MEM, self.NORMAL_NUN, self.NORMAL_PE]