import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def information_content(self, start=0, end=None, e_freq_table=None, log_base=2, chars_to_ignore=None, pseudo_count=0):
    """Calculate the information content for each residue along an alignment.

        Arguments:
         - start, end - The starting an ending points to calculate the
           information content. These points should be relative to the first
           sequence in the alignment, starting at zero (ie. even if the 'real'
           first position in the seq is 203 in the initial sequence, for
           the info content, we need to use zero). This defaults to the entire
           length of the first sequence.
         - e_freq_table - A dictionary specifying the expected frequencies
           for each letter (e.g. {'G' : 0.4, 'C' : 0.4, 'T' : 0.1, 'A' : 0.1}).
           Gap characters should not be included, since these should not have
           expected frequencies.
         - log_base - The base of the logarithm to use in calculating the
           information content. This defaults to 2 so the info is in bits.
         - chars_to_ignore - A listing of characters which should be ignored
           in calculating the info content. Defaults to none.

        Returns:
         - A number representing the info content for the specified region.

        Please see the Biopython manual for more information on how information
        content is calculated.

        """
    warnings.warn("The `information_content` method and `ic_vector` attribute of the `SummaryInfo` class are deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and use the `information_content` attribute of the Alignment obecjt. For example, for a multiple sequence alignment `msa` of DNA nucleotides, you would do: \n>>> alignment = msa.alignment\n>>> from Bio.motifs import Motif\n>>> motif = Motif('ACGT', alignment)\n>>> information_content = motif.information_content\n\nThe `information_content` object contains the same values as the `ic_vector` attribute of the `SummaryInfo` object. Its sum is equal to the value return by the `information_content` method. \nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.", BiopythonDeprecationWarning)
    if end is None:
        end = len(self.alignment[0].seq)
    if chars_to_ignore is None:
        chars_to_ignore = []
    if start < 0 or end > len(self.alignment[0].seq):
        raise ValueError('Start (%s) and end (%s) are not in the range %s to %s' % (start, end, 0, len(self.alignment[0].seq)))
    random_expected = None
    all_letters = self._get_all_letters()
    for char in chars_to_ignore:
        all_letters = all_letters.replace(char, '')
    info_content = {}
    for residue_num in range(start, end):
        freq_dict = self._get_letter_freqs(residue_num, self.alignment, all_letters, chars_to_ignore, pseudo_count, e_freq_table, random_expected)
        column_score = self._get_column_info_content(freq_dict, e_freq_table, log_base, random_expected)
        info_content[residue_num] = column_score
    total_info = sum(info_content.values())
    self.ic_vector = []
    for i, k in enumerate(info_content):
        self.ic_vector.append(info_content[i + start])
    return total_info