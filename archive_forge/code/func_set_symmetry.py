import warnings
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def set_symmetry(self, spacegroup, cell):
    """Set symmetry."""