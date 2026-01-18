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
@classmethod
def isoschizomers(cls, batch=None):
    """List all isoschizomers of the enzyme.

        Return a tuple of all the equischizomers and neoschizomers of RE.
        If batch is supplied it is used instead of the default AllEnzymes.
        """
    if not batch:
        batch = AllEnzymes
    r = [x for x in batch if cls >> x or not cls != x]
    i = r.index(cls)
    del r[i]
    r.sort()
    return r