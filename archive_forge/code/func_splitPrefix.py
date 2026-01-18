from suds.sax.enc import Encoder
def splitPrefix(name):
    """
    Split the name into a tuple (I{prefix}, I{name}). The first element in the
    tuple is I{None} when the name does not have a prefix.

    @param name: A node name containing an optional prefix.
    @type name: basestring
    @return: A tuple containing the (2) parts of I{name}.
    @rtype: (I{prefix}, I{name})

    """
    if isinstance(name, str) and ':' in name:
        return tuple(name.split(':', 1))
    return (None, name)