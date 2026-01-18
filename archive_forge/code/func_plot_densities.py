from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@add_fig_kwargs
def plot_densities(self, ax: plt.Axes=None, **kwargs):
    """
        Plot the PAW densities.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            `matplotlib` figure
        """
    ax, fig = get_ax_fig(ax)
    ax.grid(visible=True)
    ax.set_xlabel('r [Bohr]')
    for idx, density_name in enumerate(['ae_core_density', 'pseudo_core_density']):
        rden = getattr(self, density_name)
        label = '$n_c$' if idx == 1 else '$\\tilde{n}_c$'
        ax.plot(rden.mesh, rden.mesh * rden.values, label=label, lw=2)
    ax.legend(loc='best')
    return fig