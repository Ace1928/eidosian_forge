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
def show_codes(cls):
    """Print a list of supplier codes."""
    supply = [' = '.join(i) for i in cls.suppl_codes().items()]
    print('\n'.join(supply))