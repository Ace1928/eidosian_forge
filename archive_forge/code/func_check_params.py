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
def check_params(self) -> None:
    """Check INCAR for invalid tags or values.
        If a tag doesn't exist, calculation will still run, however VASP
        will ignore the tag and set it as default without letting you know.
        """
    with open(os.path.join(module_dir, 'incar_parameters.json'), encoding='utf-8') as json_file:
        incar_params = json.loads(json_file.read())
    for tag, val in self.items():
        if tag not in incar_params:
            warnings.warn(f'Cannot find {tag} in the list of INCAR tags', BadIncarWarning, stacklevel=2)
            continue
        param_type = incar_params[tag].get('type')
        allowed_values = incar_params[tag].get('values')
        if param_type is not None and type(val).__name__ != param_type:
            warnings.warn(f'{tag}: {val} is not a {param_type}', BadIncarWarning, stacklevel=2)
        if allowed_values is not None and val not in allowed_values:
            warnings.warn(f'{tag}: Cannot find {val} in the list of values', BadIncarWarning, stacklevel=2)