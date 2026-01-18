import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def sort_token_data(self, data: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    data = [(i, n, v, x) for x, (i, n, v) in enumerate(data)]
    data.sort(key=lambda x: (x[0] != self.cfg._ch10_bin_int, x[0], x[1], x[3]))
    return [(i, n, v) for i, n, v, _ in data]