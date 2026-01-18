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
def sha256_computed_file_hash(self) -> str:
    """Computes a SHA256 hash of the PotcarSingle EXCLUDING lines starting with 'SHA256' and 'COPYR'."""
    potcar_list = self.data.split('\n')
    potcar_to_hash = [line for line in potcar_list if not line.strip().startswith(('SHA256', 'COPYR'))]
    potcar_to_hash_str = '\n'.join(potcar_to_hash)
    return sha256(potcar_to_hash_str.encode('utf-8')).hexdigest()