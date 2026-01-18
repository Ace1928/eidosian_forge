from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def number_label(list_numbers):
    list_numbers = sorted(list_numbers)
    divide = [[]]
    divide[0].append(list_numbers[0])
    group = 0
    for idx in range(1, len(list_numbers)):
        if list_numbers[idx] == list_numbers[idx - 1] + 1:
            divide[group].append(list_numbers[idx])
        else:
            group += 1
            divide.append([list_numbers[idx]])
    label = ''
    for elem in divide:
        if len(elem) > 1:
            label += f'{elem[0]}-{elem[-1]},'
        else:
            label += f'{elem[0]},'
    return label[:-1]