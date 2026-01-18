from pyasn1 import error
@property
def superTags(self):
    """Return ASN.1 tags

        Returns
        -------
        : :py:class:`tuple`
            Tuple of :class:`~pyasn1.type.tag.Tag` objects that this *TagSet* contains
        """
    return self.__superTags