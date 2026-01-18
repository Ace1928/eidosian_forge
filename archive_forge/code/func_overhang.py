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
def overhang(cls):
    """Return the type of the enzyme's overhang as string.

        Can be "3' overhang", "5' overhang", "blunt", "unknown".
        """
    return "3' overhang"