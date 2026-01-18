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
def is_isoschizomer(cls, other):
    """Test for same recognition site.

        True if other has the same recognition site, else False.

        Isoschizomer: same site.

        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI
        >>> SacI.is_isoschizomer(SstI)
        True
        >>> SmaI.is_isoschizomer(XmaI)
        True

        """
    return not cls != other or cls >> other