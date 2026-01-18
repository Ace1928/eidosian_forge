import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def prog_data_to_token_data(self, program: int, channel: int, note: int, velocity: float) -> Optional[Tuple[int, int, int]]:
    if channel == 9:
        if self.cfg._ch10_bin_int == -1:
            return None
        return (self.cfg._ch10_bin_int, note, self.velocity_to_bin(velocity))
    instrument_bin = self.cfg._instrument_int_to_bin_int[program]
    if instrument_bin != -1:
        return (instrument_bin, note, self.velocity_to_bin(velocity))
    return None