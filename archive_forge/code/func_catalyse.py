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
def catalyse(cls, dna, linear=True):
    """List the sequence fragments after cutting dna with enzyme.

        Return a tuple of dna as will be produced by using RE to restrict the
        dna.

        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.

        If linear is False, the sequence is considered to be circular and the
        output will be modified accordingly.
        """
    r = cls.search(dna, linear)
    d = cls.dna
    if not r:
        return (d[1:],)
    fragments = []
    length = len(r) - 1
    if d.is_linear():
        fragments.append(d[1:r[0]])
        if length:
            fragments += [d[r[x]:r[x + 1]] for x in range(length)]
        fragments.append(d[r[-1]:])
    else:
        fragments.append(d[r[-1]:] + d[1:r[0]])
        if not length:
            return tuple(fragments)
        fragments += [d[r[x]:r[x + 1]] for x in range(length)]
    return tuple(fragments)