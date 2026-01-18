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
def is_neoschizomer(cls, other):
    """Test for neoschizomer.

        True if other is an isoschizomer of RE, else False.
        Neoschizomer: same site, different position of restriction.
        """
    return cls >> other