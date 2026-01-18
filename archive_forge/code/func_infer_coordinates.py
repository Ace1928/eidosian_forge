import sys
import collections
import copy
import importlib
import types
import warnings
import numbers
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Dict
from Bio.Align import _pairwisealigner  # type: ignore
from Bio.Align import _codonaligner  # type: ignore
from Bio.Align import substitution_matrices
from Bio.Data import CodonTable
from Bio.Seq import Seq, MutableSeq, reverse_complement, UndefinedSequenceError
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord, _RestrictedDict
@classmethod
def infer_coordinates(cls, lines, skipped_columns=None):
    """Infer the coordinates from a printed alignment.

        This method is primarily employed in Biopython's alignment parsers,
        though it may be useful for other purposes.

        For an alignment consisting of N sequences, printed as N lines with
        the same number of columns, where gaps are represented by dashes,
        this method will calculate the sequence coordinates that define the
        alignment. The coordinates are returned as a NumPy array of integers,
        and can be used to create an Alignment object.

        The argument skipped columns should be None (the default) or an empty
        list. If skipped_columns is a list, then the indices of any columns in
        the alignment with a gap in all lines are appended to skipped_columns.

        This is an example for the alignment of three sequences TAGGCATACGTG,
        AACGTACGT, and ACGCATACTTG, with gaps in the second and third sequence:

        >>> from Bio.Align import Alignment
        >>> lines = ["TAGGCATACGTG",
        ...          "AACG--TACGT-",
        ...          "-ACGCATACTTG",
        ...         ]
        >>> sequences = [line.replace("-", "") for line in lines]
        >>> sequences
        ['TAGGCATACGTG', 'AACGTACGT', 'ACGCATACTTG']
        >>> coordinates = Alignment.infer_coordinates(lines)
        >>> coordinates
        array([[ 0,  1,  4,  6, 11, 12],
               [ 0,  1,  4,  4,  9,  9],
               [ 0,  0,  3,  5, 10, 11]])
        >>> alignment = Alignment(sequences, coordinates)
        """
    n = len(lines)
    m = len(lines[0])
    for line in lines:
        assert m == len(line)
    path = []
    if m > 0:
        indices = [0] * n
        current_state = [None] * n
        for i in range(m):
            next_state = [line[i] != '-' for line in lines]
            if not any(next_state):
                if skipped_columns is not None:
                    skipped_columns.append(i)
            elif next_state == current_state:
                step += 1
            else:
                indices = [index + step if state else index for index, state in zip(indices, current_state)]
                path.append(indices)
                step = 1
                current_state = next_state
        indices = [index + step if state else index for index, state in zip(indices, current_state)]
        path.append(indices)
    coordinates = np.array(path).transpose()
    return coordinates