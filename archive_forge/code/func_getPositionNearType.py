import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def getPositionNearType(self, tagSet, idx):
    """Return the closest field position where given ASN.1 type is allowed.

        Some ASN.1 serialisation allow for skipping optional and defaulted fields.
        Some constructed ASN.1 types allow reordering of the fields. When recovering
        such objects it may be important to know at which field position, in field set,
        given *tagSet* is allowed at or past *idx* position.

        Parameters
        ----------
        tagSet: :class:`~pyasn1.type.tag.TagSet`
           ASN.1 type which field position to look up

        idx: :py:class:`int`
            Field position at or past which to perform ASN.1 type look up

        Returns
        -------
        : :py:class:`int`
            Field position in fields set

        Raises
        ------
        : :class:`~pyasn1.error.PyAsn1Error`
            If *tagSet* is not present or not unique within callee *NamedTypes*
            or *idx* is out of fields range
        """
    try:
        return idx + self.__ambiguousTypes[idx].getPositionByType(tagSet)
    except KeyError:
        raise error.PyAsn1Error('Type position out of range')