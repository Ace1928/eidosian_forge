from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def fill_region(regions, note, rect, cutoff):
    """Fill the region defined by rect with a (note, velocity, 0) color

    The velocity varies from a small value at the top of the region to
    127 at the bottom. The vertical region 0 to cutoff is split into
    three parts, with velocities 42, 84 and 127. Everything below cutoff
    has velocity 127.

    """
    x, y, width, height = rect
    if cutoff is None:
        cutoff = height
    delta_height = cutoff // 3
    regions.fill((note, 42, 0), (x, y, width, delta_height))
    regions.fill((note, 84, 0), (x, y + delta_height, width, delta_height))
    regions.fill((note, 127, 0), (x, y + 2 * delta_height, width, height - 2 * delta_height))