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
def alignment(self):
    """Return an Alignment object based on the MultipleSeqAlignment object.

        This makes a copy of each SeqRecord with a gap-less sequence. Any
        future changes to the original records in the MultipleSeqAlignment will
        not affect the new records in the Alignment.
        """
    records = [copy.copy(record) for record in self._records]
    if records:
        lines = [str(record.seq) for record in records]
        coordinates = Alignment.infer_coordinates(lines)
        for record in records:
            if record.letter_annotations:
                indices = [i for i, c in enumerate(record.seq) if c != '-']
                letter_annotations = dict(record.letter_annotations)
                record.letter_annotations.clear()
                record.seq = record.seq.replace('-', '')
                for key, value in letter_annotations.items():
                    if isinstance(value, str):
                        value = ''.join([value[i] for i in indices])
                    else:
                        cls = type(value)
                        value = cls((value[i] for i in indices))
                    letter_annotations[key] = value
                record.letter_annotations = letter_annotations
            else:
                record.seq = record.seq.replace('-', '')
        alignment = Alignment(records, coordinates)
    else:
        alignment = Alignment([])
    alignment.annotations = self.annotations
    alignment.column_annotations = self.column_annotations
    return alignment