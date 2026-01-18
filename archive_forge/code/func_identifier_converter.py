from collections import namedtuple
def identifier_converter(self, name):
    """
        Apply a conversion to the identifier for the purposes of comparison.

        The default identifier converter is for case sensitive comparison.
        """
    return name