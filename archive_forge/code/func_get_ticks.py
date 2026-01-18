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
def get_ticks(self):
    """Get all ticks and labels for a band structure plot.

        Returns:
            dict: A dictionary with 'distance': a list of distance at which
            ticks should be set and 'label': a list of label for each of those
            ticks.
        """
    bs = self._bs[0] if isinstance(self._bs, list) else self._bs
    ticks, distance = ([], [])
    for br in bs.branches:
        start, end = (br['start_index'], br['end_index'])
        labels = br['name'].split('-')
        if labels[0] == labels[1]:
            continue
        for idx, label in enumerate(labels):
            if label.startswith('\\') or '_' in label:
                labels[idx] = f'${label}$'
        if ticks and labels[0] != ticks[-1]:
            ticks[-1] += f'$\\mid${labels[0]}'
            ticks.append(labels[1])
            distance.append(bs.distance[end])
        else:
            ticks.extend(labels)
            distance.extend([bs.distance[start], bs.distance[end]])
    return {'distance': distance, 'label': ticks}