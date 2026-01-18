import collections
import platform
import sys
def include_extras(self, extras):
    """Include extra portions of the User-Agent.

        :param list extras:
            list of tuples of extra-name and extra-version
        """
    if any((len(extra) != 2 for extra in extras)):
        raise ValueError('Extras should be a sequence of two item tuples.')
    self._pieces.extend(extras)
    return self