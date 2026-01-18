import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def replacement_dictionary(self, skip_chars=None, letters=None):
    """Generate a replacement dictionary to plug into a substitution matrix.

        This should look at an alignment, and be able to generate the number
        of substitutions of different residues for each other in the
        aligned object.

        Will then return a dictionary with this information::

            {('A', 'C') : 10, ('C', 'A') : 12, ('G', 'C') : 15 ....}

        This also treats weighted sequences. The following example shows how
        we calculate the replacement dictionary. Given the following
        multiple sequence alignment::

            GTATC  0.5
            AT--C  0.8
            CTGTC  1.0

        For the first column we have::

            ('A', 'G') : 0.5 * 0.8 = 0.4
            ('C', 'G') : 0.5 * 1.0 = 0.5
            ('A', 'C') : 0.8 * 1.0 = 0.8

        We then continue this for all of the columns in the alignment, summing
        the information for each substitution in each column, until we end
        up with the replacement dictionary.

        Arguments:
         - skip_chars - Not used; setting it to anything other than None
           will raise a ValueError
         - letters - An iterable (e.g. a string or list of characters to include.
        """
    warnings.warn('The `replacement_dictionary` method is deprecated and will be removed in a future release of Biopython. As an alternative, you can convert the multiple sequence alignment object to a new-style Alignment object by via its `.alignment` property, and then use the `.substitutions` property  of the `Alignment` object. For example, for a multiple sequence alignment `msa` of DNA nucleotides, you would do: \n>>> alignment = msa.alignment\n>>> dictionary = alignment.substitutions\n\nIf your multiple sequence alignment object was obtained using Bio.AlignIO, then you can obtain a new-style Alignment object directly by using Bio.Align.read instead of Bio.AlignIO.read, or Bio.Align.parse instead of Bio.AlignIO.parse.', BiopythonDeprecationWarning)
    if skip_chars is not None:
        raise ValueError("argument skip_chars has been deprecated; instead, please use 'letters' to specify the characters you want to include")
    rep_dict = {(letter1, letter2): 0 for letter1 in letters for letter2 in letters}
    for rec_num1 in range(len(self.alignment)):
        for rec_num2 in range(rec_num1 + 1, len(self.alignment)):
            self._pair_replacement(self.alignment[rec_num1].seq, self.alignment[rec_num2].seq, self.alignment[rec_num1].annotations.get('weight', 1.0), self.alignment[rec_num2].annotations.get('weight', 1.0), rep_dict, letters)
    return rep_dict