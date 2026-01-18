import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def gap_consensus(self, threshold=0.7, ambiguous='X', require_multiple=False):
    """Output a fast consensus sequence of the alignment, allowing gaps.

        Same as dumb_consensus(), but allows gap on the output.

        Things to do:
         - Let the user define that with only one gap, the result
           character in consensus is gap.
         - Let the user select gap character, now
           it takes the same as input.

        """
    warnings.warn("The `gap_consensus` method is deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and then create a Motif object. You can then use the `.consensus` or `.degenerate_consensus` property of the Motif object to get a consensus sequence. For more control over how the consensus sequence is calculated, you can call the `calculate_consensus` method on the `.counts` property of the Motif object. This is an example for a multiple sequence alignment `msa` of DNA nucleotides:\n>>> from Bio.Seq import Seq\n>>> from Bio.SeqRecord import SeqRecord\n>>> from Bio.Align import MultipleSeqAlignment\n>>> from Bio.Align.AlignInfo import SummaryInfo\n>>> msa = MultipleSeqAlignment([SeqRecord(Seq('ACGT')),\n...                             SeqRecord(Seq('AT-T')),\n...                             SeqRecord(Seq('CT-T')),\n...                             SeqRecord(Seq('GT-T'))])\n>>> summary = SummaryInfo(msa)\n>>> gap_consensus = summary.gap_consensus(ambiguous='N')\n>>> print(gap_consensus)\nNT-T\n>>> alignment = msa.alignment\n>>> from Bio.motifs import Motif\n>>> motif = Motif('ACGT-', alignment)  # include '-' in alphabet\n>>> print(motif.consensus)\nAT-T\n>>> print(motif.degenerate_consensus)\nVT-T\n>>> counts = motif.counts\n>>> consensus = counts.calculate_consensus(identity=0.7)\n>>> print(consensus)\nNT-T\n\nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.", BiopythonDeprecationWarning)
    consensus = ''
    con_len = self.alignment.get_alignment_length()
    for n in range(con_len):
        atom_dict = Counter()
        num_atoms = 0
        for record in self.alignment:
            try:
                c = record[n]
            except IndexError:
                continue
            atom_dict[c] += 1
            num_atoms += 1
        max_atoms = []
        max_size = 0
        for atom in atom_dict:
            if atom_dict[atom] > max_size:
                max_atoms = [atom]
                max_size = atom_dict[atom]
            elif atom_dict[atom] == max_size:
                max_atoms.append(atom)
        if require_multiple and num_atoms == 1:
            consensus += ambiguous
        elif len(max_atoms) == 1 and max_size / num_atoms >= threshold:
            consensus += max_atoms[0]
        else:
            consensus += ambiguous
    return Seq(consensus)