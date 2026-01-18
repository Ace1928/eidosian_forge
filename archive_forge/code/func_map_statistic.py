from __future__ import annotations
import typing
from copy import copy, deepcopy
from typing import Iterable, List, cast, overload
import pandas as pd
from ._utils import array_kind, check_required_aesthetics, ninteraction
from .exceptions import PlotnineError
from .mapping.aes import NO_GROUP, SCALED_AESTHETICS, aes
from .mapping.evaluation import evaluate, stage
def map_statistic(self, plot: ggplot):
    for l in self:
        l.map_statistic(plot)