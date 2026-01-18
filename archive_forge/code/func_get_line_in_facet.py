from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def get_line_in_facet(self, facet):
    """Returns the sorted pts in a facet used to draw a line."""
    lines = list(facet.outer_lines)
    pt = []
    prev = None
    while len(lines) > 0:
        if prev is None:
            line = lines.pop(0)
        else:
            for idx, line in enumerate(lines):
                if prev in line:
                    line = lines.pop(idx)
                    if line[1] == prev:
                        line.reverse()
                    break
        pt.extend((self.wulff_pt_list[line[0]].tolist(), self.wulff_pt_list[line[1]].tolist()))
        prev = line[1]
    return pt