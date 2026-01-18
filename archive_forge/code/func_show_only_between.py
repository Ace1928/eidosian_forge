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
def show_only_between(self, start, end, dct=None):
    """Return only results from within start, end.

        Enzymes must cut inside start/end and may also cut outside. However,
        only the cutting positions within start/end will be returned.
        """
    d = []
    if start <= end:
        d = [(k, [vv for vv in v if start <= vv <= end]) for k, v in self.between(start, end, dct).items()]
    else:
        d = [(k, [vv for vv in v if start <= vv or vv <= end]) for k, v in self.between(start, end, dct).items()]
    return dict(d)