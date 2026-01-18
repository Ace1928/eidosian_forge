import warnings
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def init_residue(self, resname, field, resseq, icode):
    """Create a new Residue object.

        Arguments:
         - resname - string, e.g. "ASN"
         - field - hetero flag, "W" for waters, "H" for
           hetero residues, otherwise blank.
         - resseq - int, sequence identifier
         - icode - string, insertion code

        """
    if field != ' ':
        if field == 'H':
            field = 'H_' + resname
    res_id = (field, resseq, icode)
    if field == ' ':
        if self.chain.has_id(res_id):
            warnings.warn("WARNING: Residue ('%s', %i, '%s') redefined at line %i." % (field, resseq, icode, self.line_counter), PDBConstructionWarning)
            duplicate_residue = self.chain[res_id]
            if duplicate_residue.is_disordered() == 2:
                if duplicate_residue.disordered_has_id(resname):
                    self.residue = duplicate_residue
                    duplicate_residue.disordered_select(resname)
                else:
                    new_residue = Residue(res_id, resname, self.segid)
                    duplicate_residue.disordered_add(new_residue)
                    self.residue = duplicate_residue
                    return
            else:
                if resname == duplicate_residue.resname:
                    warnings.warn("WARNING: Residue ('%s', %i, '%s','%s') already defined with the same name at line  %i." % (field, resseq, icode, resname, self.line_counter), PDBConstructionWarning)
                    self.residue = duplicate_residue
                    return
                if not self._is_completely_disordered(duplicate_residue):
                    self.residue = None
                    raise PDBConstructionException("Blank altlocs in duplicate residue %s ('%s', %i, '%s')" % (resname, field, resseq, icode))
                self.chain.detach_child(res_id)
                new_residue = Residue(res_id, resname, self.segid)
                disordered_residue = DisorderedResidue(res_id)
                self.chain.add(disordered_residue)
                disordered_residue.disordered_add(duplicate_residue)
                disordered_residue.disordered_add(new_residue)
                self.residue = disordered_residue
                return
    self.residue = Residue(res_id, resname, self.segid)
    self.chain.add(self.residue)