from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def fullcompare(self, other):
    """Compare two names, returning a 3-tuple
        ``(relation, order, nlabels)``.

        *relation* describes the relation ship between the names,
        and is one of: ``dns.name.NAMERELN_NONE``,
        ``dns.name.NAMERELN_SUPERDOMAIN``, ``dns.name.NAMERELN_SUBDOMAIN``,
        ``dns.name.NAMERELN_EQUAL``, or ``dns.name.NAMERELN_COMMONANCESTOR``.

        *order* is < 0 if *self* < *other*, > 0 if *self* > *other*, and ==
        0 if *self* == *other*.  A relative name is always less than an
        absolute name.  If both names have the same relativity, then
        the DNSSEC order relation is used to order them.

        *nlabels* is the number of significant labels that the two names
        have in common.

        Here are some examples.  Names ending in "." are absolute names,
        those not ending in "." are relative names.

        =============  =============  ===========  =====  =======
        self           other          relation     order  nlabels
        =============  =============  ===========  =====  =======
        www.example.   www.example.   equal        0      3
        www.example.   example.       subdomain    > 0    2
        example.       www.example.   superdomain  < 0    2
        example1.com.  example2.com.  common anc.  < 0    2
        example1       example2.      none         < 0    0
        example1.      example2       none         > 0    0
        =============  =============  ===========  =====  =======
        """
    sabs = self.is_absolute()
    oabs = other.is_absolute()
    if sabs != oabs:
        if sabs:
            return (NAMERELN_NONE, 1, 0)
        else:
            return (NAMERELN_NONE, -1, 0)
    l1 = len(self.labels)
    l2 = len(other.labels)
    ldiff = l1 - l2
    if ldiff < 0:
        l = l1
    else:
        l = l2
    order = 0
    nlabels = 0
    namereln = NAMERELN_NONE
    while l > 0:
        l -= 1
        l1 -= 1
        l2 -= 1
        label1 = self.labels[l1].lower()
        label2 = other.labels[l2].lower()
        if label1 < label2:
            order = -1
            if nlabels > 0:
                namereln = NAMERELN_COMMONANCESTOR
            return (namereln, order, nlabels)
        elif label1 > label2:
            order = 1
            if nlabels > 0:
                namereln = NAMERELN_COMMONANCESTOR
            return (namereln, order, nlabels)
        nlabels += 1
    order = ldiff
    if ldiff < 0:
        namereln = NAMERELN_SUPERDOMAIN
    elif ldiff > 0:
        namereln = NAMERELN_SUBDOMAIN
    else:
        namereln = NAMERELN_EQUAL
    return (namereln, order, nlabels)