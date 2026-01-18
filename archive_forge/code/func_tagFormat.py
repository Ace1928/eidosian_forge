from pyasn1 import error
@property
def tagFormat(self):
    """ASN.1 tag format

        Returns
        -------
        : :py:class:`int`
            Tag format
        """
    return self.__tagFormat