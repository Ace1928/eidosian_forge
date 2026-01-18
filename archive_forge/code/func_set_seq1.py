from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def set_seq1(self, a):
    """Set the first sequence to be compared.

        The second sequence to be compared is not changed.

        >>> s = SequenceMatcher(None, "abcd", "bcde")
        >>> s.ratio()
        0.75
        >>> s.set_seq1("bcde")
        >>> s.ratio()
        1.0
        >>>

        SequenceMatcher computes and caches detailed information about the
        second sequence, so if you want to compare one sequence S against
        many sequences, use .set_seq2(S) once and call .set_seq1(x)
        repeatedly for each of the other sequences.

        See also set_seqs() and set_seq2().
        """
    if a is self.a:
        return
    self.a = a
    self.matching_blocks = self.opcodes = None