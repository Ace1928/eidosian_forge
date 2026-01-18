import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
def lambdasplit(self, func):
    """Filter enzymes in batch with supplied function.

        The new batch will contain only the enzymes for which
        func return True.
        """
    d = list(filter(func, self))
    new = RestrictionBatch()
    new._data = dict(zip(d, [True] * len(d)))
    return new