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
def get_ticks_old(self):
    """Get all ticks and labels for a band structure plot.

        Returns:
            dict: A dictionary with 'distance': a list of distance at which
            ticks should be set and 'label': a list of label for each of those
            ticks.
        """
    bs = self._bs[0]
    tick_distance = []
    tick_labels = []
    previous_label = bs.kpoints[0].label
    previous_branch = bs.branches[0]['name']
    for idx, kpt in enumerate(bs.kpoints):
        if kpt.label is not None:
            tick_distance.append(bs.distance[idx])
            this_branch = None
            for b in bs.branches:
                if b['start_index'] <= idx <= b['end_index']:
                    this_branch = b['name']
                    break
            if kpt.label != previous_label and previous_branch != this_branch:
                label1 = kpt.label
                if label1.startswith('\\') or label1.find('_') != -1:
                    label1 = f'${label1}$'
                label0 = previous_label
                if label0.startswith('\\') or label0.find('_') != -1:
                    label0 = f'${label0}$'
                tick_labels.pop()
                tick_distance.pop()
                tick_labels.append(label0 + '$\\mid$' + label1)
            elif kpt.label.startswith('\\') or kpt.label.find('_') != -1:
                tick_labels.append(f'${kpt.label}$')
            else:
                tick_labels.append(kpt.label)
            previous_label = kpt.label
            previous_branch = this_branch
    return {'distance': tick_distance, 'label': tick_labels}