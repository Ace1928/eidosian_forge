from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def softmatch(self, other):
    """
        Soft matching to see if two potentials match.

        Will only match those attributes which *are* defined for this basis info object (one way checking)
        """
    if not isinstance(other, PotentialInfo):
        return False
    d1 = self.as_dict()
    d2 = other.as_dict()
    return all((not (v is not None and v != d2[k]) for k, v in d1.items()))