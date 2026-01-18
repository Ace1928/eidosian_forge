from pyasn1 import error
@property
def tagClass(self):
    """ASN.1 tag class

        Returns
        -------
        : :py:class:`int`
            Tag class
        """
    return self.__tagClass