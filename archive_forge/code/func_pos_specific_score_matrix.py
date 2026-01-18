import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def pos_specific_score_matrix(self, axis_seq=None, chars_to_ignore=None):
    """Create a position specific score matrix object for the alignment.

        This creates a position specific score matrix (pssm) which is an
        alternative method to look at a consensus sequence.

        Arguments:
         - chars_to_ignore - A list of all characters not to include in
           the pssm.
         - axis_seq - An optional argument specifying the sequence to
           put on the axis of the PSSM. This should be a Seq object. If nothing
           is specified, the consensus sequence, calculated with default
           parameters, will be used.

        Returns:
         - A PSSM (position specific score matrix) object.

        """
    warnings.warn("The `pos_specific_score_matrix` method is deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and then create a Motif object. For example, for a multiple sequence alignment `msa` of DNA nucleotides, you would do: \n>>> alignment = msa.alignment\n>>> from Bio.motifs import Motif\n>>> motif = Motif('ACGT', alignment)\n>>> counts = motif.counts\n\nThe `counts` object contains the same information as the PSSM returned by `pos_specific_score_matrix`, but note that the indices are reversed:\n\n>>> counts[letter][i] == pssm[index][letter]\nTrue\n\nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.", BiopythonDeprecationWarning)
    all_letters = self._get_all_letters()
    if not all_letters:
        raise ValueError('_get_all_letters returned empty string')
    if chars_to_ignore is None:
        chars_to_ignore = []
    if not isinstance(chars_to_ignore, list):
        raise TypeError('chars_to_ignore should be a list.')
    gap_char = '-'
    chars_to_ignore.append(gap_char)
    for char in chars_to_ignore:
        all_letters = all_letters.replace(char, '')
    if axis_seq:
        left_seq = axis_seq
        if len(axis_seq) != self.alignment.get_alignment_length():
            raise ValueError('Axis sequence length does not equal the get_alignment_length')
    else:
        left_seq = self.dumb_consensus()
    pssm_info = []
    for residue_num in range(len(left_seq)):
        score_dict = dict.fromkeys(all_letters, 0)
        for record in self.alignment:
            try:
                this_residue = record.seq[residue_num]
            except IndexError:
                this_residue = None
            if this_residue and this_residue not in chars_to_ignore:
                weight = record.annotations.get('weight', 1.0)
                try:
                    score_dict[this_residue] += weight
                except KeyError:
                    raise ValueError('Residue %s not found' % this_residue) from None
        pssm_info.append((left_seq[residue_num], score_dict))
    return PSSM(pssm_info)