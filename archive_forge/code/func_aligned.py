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
@property
def aligned(self):
    """Return the indices of subsequences aligned to each other.

        This property returns the start and end indices of subsequences
        in the target and query sequence that were aligned to each other.
        If the alignment between target (t) and query (q) consists of N
        chunks, you get two tuples of length N:

            (((t_start1, t_end1), (t_start2, t_end2), ..., (t_startN, t_endN)),
             ((q_start1, q_end1), (q_start2, q_end2), ..., (q_startN, q_endN)))

        For example,

        >>> from Bio import Align
        >>> aligner = Align.PairwiseAligner()
        >>> alignments = aligner.align("GAACT", "GAT")
        >>> alignment = alignments[0]
        >>> print(alignment)
        target            0 GAACT 5
                          0 ||--| 5
        query             0 GA--T 3
        <BLANKLINE>
        >>> alignment.aligned
        array([[[0, 2],
                [4, 5]],
        <BLANKLINE>
               [[0, 2],
                [2, 3]]])
        >>> alignment = alignments[1]
        >>> print(alignment)
        target            0 GAACT 5
                          0 |-|-| 5
        query             0 G-A-T 3
        <BLANKLINE>
        >>> alignment.aligned
        array([[[0, 1],
                [2, 3],
                [4, 5]],
        <BLANKLINE>
               [[0, 1],
                [1, 2],
                [2, 3]]])

        Note that different alignments may have the same subsequences
        aligned to each other. In particular, this may occur if alignments
        differ from each other in terms of their gap placement only:

        >>> aligner.mismatch_score = -10
        >>> alignments = aligner.align("AAACAAA", "AAAGAAA")
        >>> len(alignments)
        2
        >>> print(alignments[0])
        target            0 AAAC-AAA 7
                          0 |||--||| 8
        query             0 AAA-GAAA 7
        <BLANKLINE>
        >>> alignments[0].aligned
        array([[[0, 3],
                [4, 7]],
        <BLANKLINE>
               [[0, 3],
                [4, 7]]])
        >>> print(alignments[1])
        target            0 AAA-CAAA 7
                          0 |||--||| 8
        query             0 AAAG-AAA 7
        <BLANKLINE>
        >>> alignments[1].aligned
        array([[[0, 3],
                [4, 7]],
        <BLANKLINE>
               [[0, 3],
                [4, 7]]])

        The property can be used to identify alignments that are identical
        to each other in terms of their aligned sequences.
        """
    if len(self.sequences) > 2:
        raise NotImplementedError('aligned is currently implemented for pairwise alignments only')
    coordinates = self.coordinates.copy()
    steps = np.diff(coordinates, 1)
    aligned = sum(steps != 0, 0) > 1
    for i, sequence in enumerate(self.sequences):
        row = steps[i, aligned]
        if (row >= 0).all():
            pass
        elif (row <= 0).all():
            steps[i, :] = -steps[i, :]
            coordinates[i, :] = len(sequence) - coordinates[i, :]
        else:
            raise ValueError(f'Inconsistent steps in row {i}')
    coordinates = coordinates.transpose()
    steps = np.diff(coordinates, axis=0)
    steps = abs(steps).min(1)
    indices = np.flatnonzero(steps)
    starts = coordinates[indices, :]
    ends = coordinates[indices + 1, :]
    segments = np.stack([starts, ends], axis=0).transpose()
    steps = np.diff(self.coordinates, 1)
    for i, sequence in enumerate(self.sequences):
        row = steps[i, aligned]
        if (row >= 0).all():
            pass
        elif (row <= 0).all():
            segments[i, :] = len(sequence) - segments[i, :]
        else:
            raise ValueError(f'Inconsistent steps in row {i}')
    return segments