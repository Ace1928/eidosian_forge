from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def batch_write_input(structures, vasp_input_set=MPRelaxSet, output_dir='.', make_dir_if_not_present=True, subfolder=None, sanitize=False, include_cif=False, potcar_spec=False, zip_output=False, **kwargs):
    """
    Batch write vasp input for a sequence of structures to
    output_dir, following the format output_dir/{group}/{formula}_{number}.

    Args:
        structures ([Structure]): Sequence of Structures.
        vasp_input_set (VaspInputSet): VaspInputSet class that creates
            vasp input files from structures. Note that a class should be
            supplied. Defaults to MPRelaxSet.
        output_dir (str): Directory to output files. Defaults to current
            directory ".".
        make_dir_if_not_present (bool): Create the directory if not present.
            Defaults to True.
        subfolder (callable): Function to create subdirectory name from
            structure. Defaults to simply "formula_count".
        sanitize (bool): Boolean indicating whether to sanitize the
            structure before writing the VASP input files. Sanitized output
            are generally easier for viewing and certain forms of analysis.
            Defaults to False.
        include_cif (bool): Whether to output a CIF as well. CIF files are
            generally better supported in visualization programs.
        potcar_spec (bool): Instead of writing the POTCAR, write a "POTCAR.spec".
                This is intended to help sharing an input set with people who might
                not have a license to specific Potcar files. Given a "POTCAR.spec",
                the specific POTCAR file can be re-generated using pymatgen with the
                "generate_potcar" function in the pymatgen CLI.
        zip_output (bool): If True, output will be zipped into a file with the
            same name as the InputSet (e.g., MPStaticSet.zip)
        **kwargs: Additional kwargs are passed to the vasp_input_set class
            in addition to structure.
    """
    output_dir = Path(output_dir)
    for idx, site in enumerate(structures):
        formula = re.sub('\\s+', '', site.formula)
        if subfolder is not None:
            subdir = subfolder(site)
            d = output_dir / subdir
        else:
            d = output_dir / f'{formula}_{idx}'
        if sanitize:
            site = site.copy(sanitize=True)
        v = vasp_input_set(site, **kwargs)
        v.write_input(str(d), make_dir_if_not_present=make_dir_if_not_present, include_cif=include_cif, potcar_spec=potcar_spec, zip_output=zip_output)