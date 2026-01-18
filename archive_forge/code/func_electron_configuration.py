from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
@property
def electron_configuration(self) -> list[tuple[int, str, int]] | None:
    """Electronic configuration of the PotcarSingle."""
    if not self.nelectrons.is_integer():
        warnings.warn('POTCAR has non-integer charge, electron configuration not well-defined.')
        return None
    el = Element.from_Z(self.atomic_no)
    full_config = el.full_electronic_structure
    nelect = self.nelectrons
    config = []
    while nelect > 0:
        e = full_config.pop(-1)
        config.append(e)
        nelect -= e[-1]
    return config