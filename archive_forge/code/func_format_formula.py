from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def format_formula(formula: str) -> str:
    """Converts str of chemical formula into
    latex format for labelling purposes.

    Args:
        formula (str): Chemical formula
    """
    formatted_formula = ''
    number_format = ''
    for idx, char in enumerate(formula, start=1):
        if char.isdigit():
            if not number_format:
                number_format = '_{'
            number_format += char
            if idx == len(formula):
                number_format += '}'
                formatted_formula += number_format
        else:
            if number_format:
                number_format += '}'
                formatted_formula += number_format
                number_format = ''
            formatted_formula += char
    return f'${formatted_formula}$'