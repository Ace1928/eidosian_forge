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
def outside(self, start, end, dct=None):
    """Return only results from enzymes that at least cut outside borders.

        Enzymes that cut outside the region in between start and end.
        They may cut inside as well.
        """
    start, end, test = self._boundaries(start, end)
    if not dct:
        dct = self.mapping
    d = {}
    for key, sites in dct.items():
        for site in sites:
            if test(start, end, site):
                continue
            else:
                d[key] = sites
                break
    return d