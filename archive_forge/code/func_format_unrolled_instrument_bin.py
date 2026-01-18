import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def format_unrolled_instrument_bin(self, instrument_bin: int) -> str:
    return f'i{self.cfg.short_instr_bin_names[instrument_bin]}'