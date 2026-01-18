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
@property
def defined_ranges(self):
    """Return a tuple of the ranges where the sequence contents is defined.

        The return value has the format ((start1, end1), (start2, end2), ...).
        """
    return tuple(((start, start + len(seq)) for start, seq in self._data.items()))