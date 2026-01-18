from __future__ import annotations
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from monty.serialization import loadfn
from ruamel import yaml
from scipy.optimize import curve_fit
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.structure_analyzer import sulfide_type
from pymatgen.core import Composition, Element
def compute_from_files(self, exp_gz: str, comp_gz: str):
    """
        Args:
            exp_gz: name of .json.gz file that contains experimental data
                    data in .json.gz file should be a list of dictionary objects with the following keys/values:
                    {"formula": chemical formula, "exp energy": formation energy in eV/formula unit,
                    "uncertainty": uncertainty in formation energy}
            comp_gz: name of .json.gz file that contains computed entries
                    data in .json.gz file should be a dictionary of {chemical formula: ComputedEntry}.
        """
    exp_entries = loadfn(exp_gz)
    calc_entries = loadfn(comp_gz)
    return self.compute_corrections(exp_entries, calc_entries)