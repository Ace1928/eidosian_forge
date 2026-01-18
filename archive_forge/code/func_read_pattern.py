from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def read_pattern(self, patterns, reverse=False, terminate_on_match=False, postprocess=str):
    """
        This function originally comes from pymatgen.io.vasp.outputs Outcar class.

        General pattern reading. Uses monty's regrep method. Takes the same
        arguments.

        Args:
            patterns (dict): A dict of patterns, e.g.,
                {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"}.
            reverse (bool): Read files in reverse. Defaults to false. Useful for
                large files, esp OUTCARs, especially when used with
                terminate_on_match.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.

        Renders accessible:
            Any attribute in patterns. For example,
            {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"} will set the
            value of self.data["energy"] = [[-1234], [-3453], ...], to the
            results from regex and postprocess. Note that the returned values
            are lists of lists, because you can grep multiple items on one line.
        """
    matches = regrep(self.filename, patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=postprocess)
    for k in patterns:
        self.data[k] = [i[0] for i in matches.get(k, [])]