import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
def isSameTypeWith(self, other, matchTags=True, matchConstraints=True):
    """Examine |ASN.1| type for equality with other ASN.1 type.

        ASN.1 tags (:py:mod:`~pyasn1.type.tag`) and constraints
        (:py:mod:`~pyasn1.type.constraint`) are examined when carrying
        out ASN.1 types comparison.

        Python class inheritance relationship is NOT considered.

        Parameters
        ----------
        other: a pyasn1 type object
            Class instance representing ASN.1 type.

        Returns
        -------
        : :class:`bool`
            :class:`True` if *other* is |ASN.1| type,
            :class:`False` otherwise.
        """
    return self is other or ((not matchTags or self.tagSet == other.tagSet) and (not matchConstraints or self.subtypeSpec == other.subtypeSpec))