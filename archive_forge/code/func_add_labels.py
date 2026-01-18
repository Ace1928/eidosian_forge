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
def add_labels(auxbox: AuxTransformBox, labels: Sequence[str], ys: Sequence[float], elements: GuideElementsColorbar) -> list[Text]:
    """
    Return Texts added to the auxbox
    """
    from matplotlib.text import Text
    n = len(labels)
    sep = elements.text.margin
    texts: list[Text] = []
    props = {'ha': elements.text.ha, 'va': elements.text.va}
    if elements.is_vertical:
        if elements.text_position == 'right':
            xs = [elements.key_width + sep] * n
        else:
            xs = [-sep] * n
    else:
        xs = ys
        if elements.text_position == 'bottom':
            ys = [-sep] * n
        else:
            ys = [elements.key_width + sep] * n
    for x, y, s in zip(xs, ys, labels):
        t = Text(x, y, s, **props)
        auxbox.add_artist(t)
        texts.append(t)
    return texts