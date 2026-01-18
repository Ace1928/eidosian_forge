import sys
from pathlib import Path
from typing import Iterable, TextIO, Tuple
import click
def read_rgb_txt(filename: Path) -> Iterable[Tuple[float, float, float]]:
    with filename.open('r') as fp:
        yield from (tuple((float_rgb_to_int(float(x)) for x in row.split())) for row in fp)