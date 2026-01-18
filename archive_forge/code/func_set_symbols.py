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
def set_symbols(self, symbols: Sequence[str], functional: str | None=None, sym_potcar_map: dict[str, str] | None=None):
    """
        Initialize the POTCAR from a set of symbols. Currently, the POTCARs can
        be fetched from a location specified in .pmgrc.yaml. Use pmg config
        to add this setting.

        Args:
            symbols (list[str]): A list of element symbols
            functional (str): The functional to use. If None, the setting
                PMG_DEFAULT_FUNCTIONAL in .pmgrc.yaml is used, or if this is
                not set, it will default to PBE.
            sym_potcar_map (dict): A map of symbol:raw POTCAR string. If
                sym_potcar_map is specified, POTCARs will be generated from
                the given map data rather than the config file location.
        """
    del self[:]
    if sym_potcar_map:
        self.extend((PotcarSingle(sym_potcar_map[el]) for el in symbols))
    else:
        self.extend((PotcarSingle.from_symbol_and_functional(el, functional) for el in symbols))