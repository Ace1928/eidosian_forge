import sys
from pathlib import Path
from typing import Iterable, TextIO, Tuple
import click
def float_rgb_to_int(value: float) -> int:
    return round(value * 255)