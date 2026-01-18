from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
@classmethod
def from_chgcar(cls, structure, chgcar=None, chgcar_ref=None, user_input_settings=None, write_cml=False, write_json=True, zpsp=None) -> Self:
    """Run Critic2 in automatic mode on a supplied structure, charge
        density (chgcar) and reference charge density (chgcar_ref).

        The reason for a separate reference field is that in
        VASP, the CHGCAR charge density only contains valence
        electrons and may be missing substantial charge at
        nuclei leading to misleading results. Thus, a reference
        field is commonly constructed from the sum of AECCAR0
        and AECCAR2 which is the total charge density, but then
        the valence charge density is used for the final analysis.

        If chgcar_ref is not supplied, chgcar will be used as the
        reference field. If chgcar is not supplied, the promolecular
        charge density will be used as the reference field -- this can
        often still give useful results if only topological information
        is wanted.

        User settings is a dictionary that can contain:
        * GRADEPS, float (field units), gradient norm threshold
        * CPEPS, float (Bohr units in crystals), minimum distance between
          critical points for them to be equivalent
        * NUCEPS, same as CPEPS but specifically for nucleus critical
          points (critic2 default is dependent on grid dimensions)
        * NUCEPSH, same as NUCEPS but specifically for hydrogen nuclei
          since associated charge density can be significantly displaced
          from hydrogen nucleus
        * EPSDEGEN, float (field units), discard critical point if any
          element of the diagonal of the Hessian is below this value,
          useful for discarding points in vacuum regions
        * DISCARD, float (field units), discard critical points with field
          value below this value, useful for discarding points in vacuum
          regions
        * SEED, list of strings, strategies for seeding points, default
          is ['WS 1', 'PAIR 10'] which seeds critical points by
          sub-dividing the Wigner-Seitz cell and between every atom pair
          closer than 10 Bohr, see critic2 manual for more options

        Args:
            structure: Structure to analyze
            chgcar: Charge density to use for analysis. If None, will
                use promolecular density. Should be a Chgcar object or path (string).
            chgcar_ref: Reference charge density. If None, will use
                chgcar as reference. Should be a Chgcar object or path (string).
            user_input_settings (dict): as explained above
            write_cml (bool): Useful for debug, if True will write all
                critical points to a file 'table.cml' in the working directory
                useful for visualization
            write_json (bool): Whether to write out critical points
                and YT json. YT integration will be performed with this setting.
            zpsp (dict): Dict of element/symbol name to number of electrons
                (ZVAL in VASP pseudopotential), with which to properly augment core regions
                and calculate charge transfer. Optional.
        """
    settings = {'CPEPS': 0.1, 'SEED': ['WS', 'PAIR DIST 10']}
    if user_input_settings:
        settings.update(user_input_settings)
    input_script = ['crystal POSCAR']
    if chgcar_ref:
        input_script += ['load ref.CHGCAR id chg_ref', 'reference chg_ref']
    if chgcar:
        input_script += ['load int.CHGCAR id chg_int', 'integrable chg_int']
        if zpsp:
            zpsp_str = f' zpsp {' '.join((f'{symbol} {int(zval)}' for symbol, zval in zpsp.items()))}'
            input_script[-2] += zpsp_str
    auto = 'auto '
    for k, v in settings.items():
        if isinstance(v, list):
            for item in v:
                auto += f'{k} {item} '
        else:
            auto += f'{k} {v} '
    input_script += [auto]
    if write_cml:
        input_script += ['cpreport ../table.cml cell border graph']
    if write_json:
        input_script += ['cpreport cpreport.json']
    if write_json and chgcar:
        input_script += ['yt']
        input_script += ['yt JSON yt.json']
    input_script_str = '\n'.join(input_script)
    with ScratchDir('.'):
        structure.to(filename='POSCAR')
        if chgcar and isinstance(chgcar, VolumetricData):
            chgcar.write_file('int.CHGCAR')
        elif chgcar:
            os.symlink(chgcar, 'int.CHGCAR')
        if chgcar_ref and isinstance(chgcar_ref, VolumetricData):
            chgcar_ref.write_file('ref.CHGCAR')
        elif chgcar_ref:
            os.symlink(chgcar_ref, 'ref.CHGCAR')
        caller = cls(input_script_str)
        caller.output = Critic2Analysis(structure, stdout=caller._stdout, stderr=caller._stderr, cpreport=caller._cp_report, yt=caller._yt, zpsp=zpsp)
        return caller