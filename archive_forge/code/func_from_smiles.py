import io
import os
import pathlib
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
@classmethod
def from_smiles(cls, data: str, caption: Optional[str]=None, sanitize: bool=True, convert_to_3d_and_optimize: bool=True, mmff_optimize_molecule_max_iterations: int=200) -> 'Molecule':
    """Convert SMILES string to wandb.Molecule.

        Arguments:
            data: (string)
                SMILES string.
            caption: (string)
                Caption associated with the molecule for display
            sanitize: (bool)
                Check if the molecule is chemically reasonable by the RDKit's definition.
            convert_to_3d_and_optimize: (bool)
                Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.
                This is an expensive operation that may take a long time for complicated molecules.
            mmff_optimize_molecule_max_iterations: (int)
                Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule
        """
    rdkit_chem = util.get_module('rdkit.Chem', required='wandb.Molecule needs the rdkit-pypi package. To get it, run "pip install rdkit-pypi".')
    molecule = rdkit_chem.MolFromSmiles(data, sanitize=sanitize)
    if molecule is None:
        raise ValueError('Unable to parse the SMILES string.')
    return cls.from_rdkit(data_or_path=molecule, caption=caption, convert_to_3d_and_optimize=convert_to_3d_and_optimize, mmff_optimize_molecule_max_iterations=mmff_optimize_molecule_max_iterations)