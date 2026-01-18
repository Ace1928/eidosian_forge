import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
def count_overlap(self, sub, start=None, end=None):
    """Return an overlapping count.

        Returns an integer, the number of occurrences of substring
        argument sub in the (sub)sequence given by [start:end].
        Optional arguments start and end are interpreted as in slice
        notation.

        Arguments:
         - sub - a string or another Seq object to look for
         - start - optional integer, slice start
         - end - optional integer, slice end

        e.g.

        >>> from Bio.Seq import Seq
        >>> print(Seq("AAAA").count_overlap("AA"))
        3
        >>> print(Seq("ATATATATA").count_overlap("ATA"))
        4
        >>> print(Seq("ATATATATA").count_overlap("ATA", 3, -1))
        1

        For a non-overlapping search, use the ``count`` method:

        >>> print(Seq("AAAA").count("AA"))
        2

        Where substrings do not overlap, ``count_overlap`` behaves the same as
        the ``count`` method:

        >>> from Bio.Seq import Seq
        >>> my_seq = Seq("AAAATGA")
        >>> print(my_seq.count_overlap("A"))
        5
        >>> my_seq.count_overlap("A") == my_seq.count("A")
        True
        >>> print(my_seq.count_overlap("ATG"))
        1
        >>> my_seq.count_overlap("ATG") == my_seq.count("ATG")
        True
        >>> print(my_seq.count_overlap(Seq("AT")))
        1
        >>> my_seq.count_overlap(Seq("AT")) == my_seq.count(Seq("AT"))
        True
        >>> print(my_seq.count_overlap("AT", 2, -1))
        1
        >>> my_seq.count_overlap("AT", 2, -1) == my_seq.count("AT", 2, -1)
        True

        HOWEVER, do not use this method for such cases because the
        count() method is much for efficient.
        """
    if isinstance(sub, MutableSeq):
        sub = sub._data
    elif isinstance(sub, Seq):
        sub = bytes(sub)
    elif isinstance(sub, str):
        sub = sub.encode('ASCII')
    elif not isinstance(sub, (bytes, bytearray)):
        raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
    data = self._data
    overlap_count = 0
    while True:
        start = data.find(sub, start, end) + 1
        if start != 0:
            overlap_count += 1
        else:
            return overlap_count