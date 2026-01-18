from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
def add_segmented_colorbar(auxbox: AuxTransformBox, colors: Sequence[str], elements: GuideElementsColorbar):
    """
    Add 'non-rastered' colorbar to AuxTransformBox
    """
    from matplotlib.collections import PolyCollection
    nbreak = len(colors)
    if elements.is_vertical:
        colorbar_height = elements.key_height
        colorbar_width = elements.key_width
        linewidth = colorbar_height / nbreak
        verts = []
        x1, x2 = (0, colorbar_width)
        for i in range(nbreak):
            y1 = i * linewidth
            y2 = y1 + linewidth
            verts.append(((x1, y1), (x1, y2), (x2, y2), (x2, y1)))
    else:
        colorbar_width = elements.key_height
        colorbar_height = elements.key_width
        linewidth = colorbar_width / nbreak
        verts = []
        y1, y2 = (0, colorbar_height)
        for i in range(nbreak):
            x1 = i * linewidth
            x2 = x1 + linewidth
            verts.append(((x1, y1), (x1, y2), (x2, y2), (x2, y1)))
    coll = PolyCollection(verts, facecolors=colors, linewidth=0, antialiased=False)
    auxbox.add_artist(coll)