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
def hint_for_accuracy(self, accuracy='normal'):
    """
        Returns a Hint object with the suggested value of ecut [Ha] and
        pawecutdg [Ha] for the given accuracy.
        ecut and pawecutdg are set to zero if no hint is available.

        Args:
            accuracy: ["low", "normal", "high"]
        """
    if not self.has_dojo_report:
        return Hint(ecut=0.0, pawecutdg=0.0)
    if 'hints' in self.dojo_report:
        return Hint.from_dict(self.dojo_report['hints'][accuracy])
    if 'ppgen_hints' in self.dojo_report:
        return Hint.from_dict(self.dojo_report['ppgen_hints'][accuracy])
    return Hint(ecut=0.0, pawecutdg=0.0)