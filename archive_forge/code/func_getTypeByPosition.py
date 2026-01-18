import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def getTypeByPosition(self, idx):
    """Return ASN.1 type object by its position in fields set.

        Parameters
        ----------
        idx: :py:class:`int`
            Field index

        Returns
        -------
        :
            ASN.1 type

        Raises
        ------
        : :class:`~pyasn1.error.PyAsn1Error`
            If given position is out of fields range
        """
    try:
        return self.__namedTypes[idx].asn1Object
    except IndexError:
        raise error.PyAsn1Error('Type position out of range')