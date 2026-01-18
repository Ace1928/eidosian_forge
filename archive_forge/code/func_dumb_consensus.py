import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def dumb_consensus(self, threshold=0.7, ambiguous='X', require_multiple=False):
    """Output a fast consensus sequence of the alignment.

        This doesn't do anything fancy at all. It will just go through the
        sequence residue by residue and count up the number of each type
        of residue (ie. A or G or T or C for DNA) in all sequences in the
        alignment. If the percentage of the most common residue type is
        greater then the passed threshold, then we will add that residue type,
        otherwise an ambiguous character will be added.

        This could be made a lot fancier (ie. to take a substitution matrix
        into account), but it just meant for a quick and dirty consensus.

        Arguments:
         - threshold - The threshold value that is required to add a particular
           atom.
         - ambiguous - The ambiguous character to be added when the threshold is
           not reached.
         - require_multiple - If set as True, this will require that more than
           1 sequence be part of an alignment to put it in the consensus (ie.
           not just 1 sequence and gaps).

        """
    warnings.warn("The `dumb_consensus` method is deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and then create a Motif object. You can then use the `.consensus` or `.degenerate_consensus` property of the Motif object to get a consensus sequence. For more control over how the consensus sequence is calculated, you can call the `calculate_consensus` method on the `.counts` property of the Motif object. This is an example for a multiple sequence alignment `msa` of DNA nucleotides:\n>>> from Bio.Seq import Seq\n>>> from Bio.SeqRecord import SeqRecord\n>>> from Bio.Align import MultipleSeqAlignment\n>>> from Bio.Align.AlignInfo import SummaryInfo\n>>> msa = MultipleSeqAlignment([SeqRecord(Seq('ACGT')),\n...                             SeqRecord(Seq('ATGT')),\n...                             SeqRecord(Seq('ATGT'))])\n>>> summary = SummaryInfo(msa)\n>>> dumb_consensus = summary.dumb_consensus(ambiguous='N')\n>>> print(dumb_consensus)\nANGT\n>>> alignment = msa.alignment\n>>> from Bio.motifs import Motif\n>>> motif = Motif('ACGT', alignment)\n>>> print(motif.consensus)\nATGT\n>>> print(motif.degenerate_consensus)\nAYGT\n>>> counts = motif.counts\n>>> consensus = counts.calculate_consensus(identity=0.7)\n>>> print(consensus)\nANGT\n\nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.", BiopythonDeprecationWarning)
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
            if c != '-' and c != '.':
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