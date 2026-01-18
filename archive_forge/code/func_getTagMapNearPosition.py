import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def getTagMapNearPosition(self, idx):
    """Return ASN.1 types that are allowed at or past given field position.

        Some ASN.1 serialisation allow for skipping optional and defaulted fields.
        Some constructed ASN.1 types allow reordering of the fields. When recovering
        such objects it may be important to know which types can possibly be
        present at any given position in the field sets.

        Parameters
        ----------
        idx: :py:class:`int`
            Field index

        Returns
        -------
        : :class:`~pyasn1.type.tagmap.TagMap`
            Map if ASN.1 types allowed at given field position

        Raises
        ------
        : :class:`~pyasn1.error.PyAsn1Error`
            If given position is out of fields range
        """
    try:
        return self.__ambiguousTypes[idx].tagMap
    except KeyError:
        raise error.PyAsn1Error('Type position out of range')