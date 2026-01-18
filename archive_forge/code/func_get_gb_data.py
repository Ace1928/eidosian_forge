from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def get_gb_data(self, material_id=None, pretty_formula=None, chemsys=None, sigma=None, gb_plane=None, rotation_axis=None, include_work_of_separation=False):
    """Gets grain boundary data for a material.

        Args:
            material_id (str): Materials Project material_id, e.g., 'mp-129'.
            pretty_formula (str): The formula of metals. e.g., 'Fe'
            chemsys (str): The chemical system. e.g., 'Fe-O'
            sigma (int): The sigma value of a certain type of grain boundary
            gb_plane (list of integer): The Miller index of grain boundary plane. e.g., [1, 1, 1]
            rotation_axis (list of integer): The Miller index of rotation axis. e.g.,
                [1, 0, 0], [1, 1, 0], and [1, 1, 1] Sigma value is determined by the combination of
                rotation axis and rotation angle. The five degrees of freedom (DOF) of one grain boundary
                include: rotation axis (2 DOFs), rotation angle (1 DOF), and grain boundary plane (2 DOFs).
            include_work_of_separation (bool): whether to include the work of separation
                (in unit of (J/m^2)). If you want to query the work of separation, please
                specify the material_id.

        Returns:
            A list of grain boundaries that satisfy the query conditions (sigma, gb_plane).
            Energies are given in SI units (J/m^2).
        """
    if gb_plane:
        gb_plane = ','.join((str(plane) for plane in gb_plane))
    if rotation_axis:
        rotation_axis = ','.join((str(ax) for ax in rotation_axis))
    payload = {'material_id': material_id, 'pretty_formula': pretty_formula, 'chemsys': chemsys, 'sigma': sigma, 'gb_plane': gb_plane, 'rotation_axis': rotation_axis}
    if include_work_of_separation and material_id:
        list_of_gbs = self._make_request('/grain_boundaries', payload=payload)
        for gb_dict in list_of_gbs:
            gb_energy = gb_dict['gb_energy']
            gb_plane_int = gb_dict['gb_plane']
            surface_energy = self.get_surface_data(material_id=material_id, miller_index=gb_plane_int)['surface_energy']
            work_of_sep = 2 * surface_energy - gb_energy
            gb_dict['work_of_separation'] = work_of_sep
        return list_of_gbs
    return self._make_request('/grain_boundaries', payload=payload)