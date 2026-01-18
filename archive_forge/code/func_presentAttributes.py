from typing import Set
@property
def presentAttributes(self):
    """
        An iterable containing the names of the attributes that are present in
        this sentence.

        @return: The iterable of names of present attributes.
        @rtype: iterable of C{str}
        """
    return iter(self._sentenceData)