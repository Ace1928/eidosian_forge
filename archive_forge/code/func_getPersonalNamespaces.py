from zope.interface import Interface
def getPersonalNamespaces():
    """
        Report the available personal namespaces.

        Typically there should be only one personal namespace. A common name
        for it is C{""}, and its hierarchical delimiter is usually C{"/"}.

        @rtype: iterable of two-tuples of strings
        @return: The personal namespaces and their hierarchical delimiters. If
            no namespaces of this type exist, None should be returned.
        """