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
def mapall(self, alignments):
    """Map each of the alignments to self, and return the mapped alignment."""
    factor = None
    alignments = list(alignments)
    for alignment in alignments:
        steps = np.diff(alignment.coordinates, 1)
        aligned = sum(steps != 0, 0) > 1
        steps = steps[:, aligned]
        step1, step2 = steps.sum(1)
        if step1 == step2:
            step = 1
        elif step1 == -step2:
            step = 1
        elif step2 == 3 * step1:
            step = 3
        else:
            raise ValueError(f'unexpected steps {step1}, {step2}')
        if factor is None:
            factor = step
        elif factor != step:
            raise ValueError('inconsistent step sizes in alignments')
    steps = abs(self.coordinates[:, 1:] - self.coordinates[:, :-1]).max(0).clip(0)
    coordinates = np.empty((2, len(steps) + 1), int)
    coordinates[0, 0] = 0
    coordinates[0, 1:] = factor * np.cumsum(steps)
    sequences = [Seq(None, length=coordinates[0, -1]), None]
    for i, alignment in enumerate(alignments):
        coordinates[1, :] = factor * self.coordinates[i, :]
        sequences[1] = Seq(None, length=coordinates[1, -1])
        alignment1 = Alignment(sequences, coordinates)
        coordinates2 = alignment.coordinates.copy()
        coordinates2[0, :] *= factor
        sequences2 = [sequences[1], alignment.sequences[1]]
        alignment2 = Alignment(sequences2, coordinates2)
        alignments[i] = alignment1.map(alignment2)
    coordinates = [[] for i in range(len(alignments))]
    done = False
    while done is False:
        done = True
        position = min((alignment.coordinates[0, 0] for alignment in alignments if alignment.coordinates.size))
        for i, alignment in enumerate(alignments):
            if alignment.coordinates.size == 0:
                coordinates[i].append(coordinates[i][-1])
            elif alignment.coordinates[0, 0] == position:
                coordinates[i].append(alignment.coordinates[1, 0])
                alignment.coordinates = alignment.coordinates[:, 1:]
                if alignment.coordinates.any():
                    done = False
            elif alignment.coordinates[0, 0] > position:
                if len(coordinates[i]):
                    if alignment.coordinates[1, 0] > coordinates[i][-1]:
                        step = position - previous
                    else:
                        step = 0
                    coordinates[i].append(coordinates[i][-1] + step)
                else:
                    coordinates[i].append(alignment.coordinates[1, 0])
            else:
                raise Exception
        previous = position
    sequences = [alignment.sequences[1] for alignment in alignments]
    coordinates = np.array(coordinates)
    alignment = Alignment(sequences, coordinates)
    return alignment