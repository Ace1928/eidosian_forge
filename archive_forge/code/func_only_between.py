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
def only_between(self, start, end, dct=None):
    """Return only results from enzymes that only cut within start, end."""
    start, end, test = self._boundaries(start, end)
    if not dct:
        dct = self.mapping
    d = dict(dct)
    for key, sites in dct.items():
        if not sites:
            del d[key]
            continue
        for site in sites:
            if test(start, end, site):
                continue
            else:
                del d[key]
                break
    return d