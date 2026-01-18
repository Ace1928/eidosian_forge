from typing import Optional, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
from .sbcharsetprober import SingleByteCharSetProber
@property
def charset_name(self) -> str:
    assert self._logical_prober is not None
    assert self._visual_prober is not None
    finalsub = self._final_char_logical_score - self._final_char_visual_score
    if finalsub >= self.MIN_FINAL_CHAR_DISTANCE:
        return self.LOGICAL_HEBREW_NAME
    if finalsub <= -self.MIN_FINAL_CHAR_DISTANCE:
        return self.VISUAL_HEBREW_NAME
    modelsub = self._logical_prober.get_confidence() - self._visual_prober.get_confidence()
    if modelsub > self.MIN_MODEL_DISTANCE:
        return self.LOGICAL_HEBREW_NAME
    if modelsub < -self.MIN_MODEL_DISTANCE:
        return self.VISUAL_HEBREW_NAME
    if finalsub < 0.0:
        return self.VISUAL_HEBREW_NAME
    return self.LOGICAL_HEBREW_NAME