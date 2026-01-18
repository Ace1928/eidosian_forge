import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def format_unrolled_velocity(self, velocity_bin: int) -> str:
    return f'v{velocity_bin:x}'