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
def do_not_cut(self, start, end, dct=None):
    """Return only results from enzymes that don't cut between borders."""
    if not dct:
        dct = self.mapping
    d = self.without_site()
    d.update(self.only_outside(start, end, dct))
    return d