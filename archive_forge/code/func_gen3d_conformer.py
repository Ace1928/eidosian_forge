from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def gen3d_conformer(self) -> None:
    """
        A combined method to first generate 3D structures from 0D or 2D
        structures and then find the minimum energy conformer:

        1. Use OBBuilder to create a 3D structure using rules and ring templates
        2. Do 250 steps of a steepest descent geometry optimization with the
           MMFF94 forcefield
        3. Do 200 iterations of a Weighted Rotor conformational search
           (optimizing each conformer with 25 steps of a steepest descent)
        4. Do 250 steps of a conjugate gradient geometry optimization.

        Warning from openbabel docs:
        For many applications where 100s if not 1000s of molecules need to be
        processed, gen3d is rather SLOW. Sometimes this function can cause a
        segmentation fault.
        A future version of Open Babel will provide options for slow/medium/fast
        3D structure generation which will involve different compromises
        between speed and finding the global energy minimum.
        """
    gen3d = openbabel.OBOp.FindType('Gen3D')
    gen3d.Do(self._ob_mol)