from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def map_hue(self, palette=None, order=None, norm=None, saturation=1):
    mapping = HueMapping(self, palette, order, norm, saturation)
    self._hue_map = mapping