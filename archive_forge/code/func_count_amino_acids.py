import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def count_amino_acids(self):
    """Count standard amino acids, return a dict.

        Counts the number times each amino acid is in the protein
        sequence. Returns a dictionary {AminoAcid:Number}.

        The return value is cached in self.amino_acids_content.
        It is not recalculated upon subsequent calls.
        """
    if self.amino_acids_content is None:
        prot_dic = {k: 0 for k in IUPACData.protein_letters}
        for aa in prot_dic:
            prot_dic[aa] = self.sequence.count(aa)
        self.amino_acids_content = prot_dic
    return self.amino_acids_content