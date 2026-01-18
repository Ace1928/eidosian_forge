import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def data_to_wait_tokens(self, delta_ms: float) -> List[str]:
    if delta_ms == 0.0:
        return []
    return [self.format_wait_token(i) for i in self.delta_to_wait_ids(delta_ms)]