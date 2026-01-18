import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
@property
def minTagSet(self):
    """Return the minimal TagSet among ASN.1 type in callee *NamedTypes*.

        Some ASN.1 types/serialisation protocols require ASN.1 types to be
        arranged based on their numerical tag value. The *minTagSet* property
        returns that.

        Returns
        -------
        : :class:`~pyasn1.type.tagset.TagSet`
            Minimal TagSet among ASN.1 types in callee *NamedTypes*
        """
    return self.__minTagSet