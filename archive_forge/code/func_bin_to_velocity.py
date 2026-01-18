import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def bin_to_velocity(self, bin: int) -> int:
    if self.cfg.velocity_bins_override:
        return self.cfg.velocity_bins_override[bin]
    binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
    if self.cfg.velocity_exp == 1.0:
        return max(0, ceil(bin * binsize - 1))
    else:
        return max(0, ceil(self.cfg.velocity_events * log((self.cfg.velocity_exp - 1) * binsize * bin / self.cfg.velocity_events + 1, self.cfg.velocity_exp) - 1))