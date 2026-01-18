import collections
from minerl.herobraine.hero.handler import Handler
import numpy as np
from functools import reduce
from typing import List, Tuple
from minerl.herobraine.hero.spaces import Box, Dict, Enum, MineRLSpace
def flatten_spaces(hdls: List[Handler]) -> Tuple[list, List[Tuple[str, MineRLSpace]]]:
    return ([hdl.space.flattened for hdl in hdls if hdl.space.is_flattenable()], [(hdl.to_string(), hdl.space) for hdl in hdls if not hdl.space.is_flattenable()])