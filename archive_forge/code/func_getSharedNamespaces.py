from zope.interface import Interface
def getSharedNamespaces():
    """
        Report the available shared namespaces.

        Shared namespaces do not belong to any individual user but are usually
        to one or more of them. Examples of shared namespaces might be
        C{"#news"} for a usenet gateway.

        @rtype: iterable of two-tuples of strings
        @return: The shared namespaces and their hierarchical delimiters. If no
            namespaces of this type exist, None should be returned.
        """