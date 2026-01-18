from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
def get_explanation_dict(self, entry):
    """Provides an explanation dict of the corrections that are being applied
        for a given compatibility scheme. Inspired by the "explain" methods
        in many database methodologies.

        Args:
            entry: A ComputedEntry.

        Returns:
            dict[str, str | float | list[dict[str, Union[str, float]]]: of the form
                {"Compatibility": "string",
                "Uncorrected_energy": float,
                "Corrected_energy": float,
                "correction_uncertainty:" float,
                "Corrections": [{"Name of Correction": {
                "Value": float, "Explanation": "string", "Uncertainty": float}]}
        """
    corr_entry = self.process_entry(entry)
    uncorrected_energy = (corr_entry or entry).uncorrected_energy
    corrected_energy = corr_entry.energy if corr_entry else None
    correction_uncertainty = corr_entry.correction_uncertainty if corr_entry else None
    dct = {'compatibility': type(self).__name__, 'uncorrected_energy': uncorrected_energy, 'corrected_energy': corrected_energy, 'correction_uncertainty': correction_uncertainty}
    corrections = []
    corr_dict, uncer_dict = self.get_corrections_dict(entry)
    for c in self.corrections:
        if corr_dict.get(str(c), 0) != 0 and uncer_dict.get(str(c), 0) == 0:
            uncer = np.nan
        else:
            uncer = uncer_dict.get(str(c), 0)
        cd = {'name': str(c), 'description': c.__doc__.split('Args')[0].strip(), 'value': corr_dict.get(str(c), 0), 'uncertainty': uncer}
        corrections.append(cd)
    dct['corrections'] = corrections
    return dct