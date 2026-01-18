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
def overhang5(self, dct=None):
    """Return only cuts that have 5' overhangs."""
    if not dct:
        dct = self.mapping
    return {k: v for k, v in dct.items() if k.is_5overhang()}