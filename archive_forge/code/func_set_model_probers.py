from typing import Optional, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
from .sbcharsetprober import SingleByteCharSetProber
def set_model_probers(self, logical_prober: SingleByteCharSetProber, visual_prober: SingleByteCharSetProber) -> None:
    self._logical_prober = logical_prober
    self._visual_prober = visual_prober