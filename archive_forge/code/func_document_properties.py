from __future__ import annotations
from dataclasses import dataclass, fields, field
import textwrap
from typing import Any, Callable, Union
from collections.abc import Generator
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import ndarray
from pandas import DataFrame
from matplotlib.artist import Artist
from seaborn._core.scales import Scale
from seaborn._core.properties import (
from seaborn._core.exceptions import PlotSpecError
def document_properties(mark):
    properties = [f.name for f in fields(mark) if isinstance(f.default, Mappable)]
    text = ['', '    This mark defines the following properties:', textwrap.fill(', '.join([f'|{p}|' for p in properties]), width=78, initial_indent=' ' * 8, subsequent_indent=' ' * 8)]
    docstring_lines = mark.__doc__.split('\n')
    new_docstring = '\n'.join([*docstring_lines[:2], *text, *docstring_lines[2:]])
    mark.__doc__ = new_docstring
    return mark