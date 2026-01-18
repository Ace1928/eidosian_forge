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
def is_blunt(cls):
    """Return if the enzyme produces blunt ends.

        True if the enzyme produces blunt end.

        Related methods:

        - RE.is_3overhang()
        - RE.is_5overhang()
        - RE.is_unknown()

        """
    return False