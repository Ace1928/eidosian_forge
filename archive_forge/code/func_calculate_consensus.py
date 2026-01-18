import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
def calculate_consensus(self, substitution_matrix=None, plurality=None, identity=0, setcase=None):
    """Return the consensus sequence (as a string) for the given parameters.

        This function largely follows the conventions of the EMBOSS `cons` tool.

        Arguments:
         - substitution_matrix - the scoring matrix used when comparing
           sequences. By default, it is None, in which case we simply count the
           frequency of each letter.
           Instead of the default value, you can use the substitution matrices
           available in Bio.Align.substitution_matrices. Common choices are
           BLOSUM62 (also known as EBLOSUM62) for protein, and NUC.4.4 (also
           known as EDNAFULL) for nucleotides. NOTE: This has not yet been
           implemented.
         - plurality           - threshold value for the number of positive
           matches, divided by the total count in a column, required to reach
           consensus. If substitution_matrix is None, then this argument must
           be None, and is ignored; a ValueError is raised otherwise. If
           substitution_matrix is not None, then the default value of the
           plurality is 0.5.
         - identity            - number of identities, divided by the total
           count in a column, required to define a consensus value. If the
           number of identities is less than identity * total count in a column,
           then the undefined character ('N' for nucleotides and 'X' for amino
           acid sequences) is used in the consensus sequence. If identity is
           1.0, then only columns of identical letters contribute to the
           consensus. Default value is zero.
         - setcase             - threshold for the positive matches, divided by
           the total count in a column, above which the consensus is is
           upper-case and below which the consensus is in lower-case. By
           default, this is equal to 0.5.
        """
    alphabet = self.alphabet
    if set(alphabet).union('ACGTUN-') == set('ACGTUN-'):
        undefined = 'N'
    else:
        undefined = 'X'
    if substitution_matrix is None:
        if plurality is not None:
            raise ValueError('plurality must be None if substitution_matrix is None')
        sequence = ''
        for i in range(self.length):
            maximum = 0
            total = 0
            for letter in alphabet:
                count = self[letter][i]
                total += count
                if count > maximum:
                    maximum = count
                    consensus_letter = letter
            if maximum < identity * total:
                consensus_letter = undefined
            else:
                if setcase is None:
                    setcase_threshold = total / 2
                else:
                    setcase_threshold = setcase * total
                if maximum <= setcase_threshold:
                    consensus_letter = consensus_letter.lower()
            sequence += consensus_letter
    else:
        raise NotImplementedError('calculate_consensus currently only supports substitution_matrix=None')
    return sequence