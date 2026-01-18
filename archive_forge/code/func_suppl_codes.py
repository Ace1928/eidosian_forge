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
def suppl_codes(cls):
    """Return a dictionary with supplier codes.

        Letter code for the suppliers.
        """
    supply = {k: v[0] for k, v in suppliers_dict.items()}
    return supply