import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def format_unrolled_note(self, note: int) -> str:
    return f'n{note:x}'