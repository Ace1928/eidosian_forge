from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def rotor_conformer(self, *rotor_args, algo: str='WeightedRotorSearch', forcefield: str='mmff94') -> None:
    """
        Conformer search based on several Rotor Search algorithms of openbabel.
        If the input molecule is not 3D, make3d will be called (generate 3D
        structure, add hydrogen, a quick localopt). All hydrogen atoms need
        to be made explicit.

        Args:
            rotor_args: pass args to Rotor Search in openbabel.
                for "WeightedRotorSearch": (conformers, geomSteps,
                sampleRingBonds-default False)
                for "SystematicRotorSearch": (geomSteps-default 2500,
                sampleRingBonds-default False)
                for "RandomRotorSearch": (conformers, geomSteps-default 2500,
                sampleRingBonds-default False)
            algo (str): Default is "WeightedRotorSearch". Options are
                "SystematicRotorSearch", "RandomRotorSearch", and
                "WeightedRotorSearch".
            forcefield (str): Default is mmff94. Options are 'gaff', 'ghemical',
                'mmff94', 'mmff94s', and 'uff'.
        """
    if self._ob_mol.GetDimension() != 3:
        self.make3d()
    else:
        self.add_hydrogen()
    ff = openbabel.OBForceField.FindType(forcefield)
    if ff == 0:
        warnings.warn(f"This input forcefield={forcefield!r} is not supported in openbabel. The forcefield will be reset as default 'mmff94' for now.")
        ff = openbabel.OBForceField.FindType('mmff94')
    try:
        rotor_search = getattr(ff, algo)
    except AttributeError:
        warnings.warn(f"This input conformer search algorithm {algo} is not supported in openbabel. Options are 'SystematicRotorSearch', 'RandomRotorSearch' and 'WeightedRotorSearch'. The algorithm will be reset as default 'WeightedRotorSearch' for now.")
        rotor_search = ff.WeightedRotorSearch
    rotor_search(*rotor_args)
    ff.GetConformers(self._ob_mol)