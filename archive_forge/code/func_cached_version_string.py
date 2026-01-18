import itertools
import operator
import sys
def cached_version_string(self, prefix=''):
    """Return a cached version string.

        This will return a cached version string if one is already cached,
        irrespective of prefix. If none is cached, one will be created with
        prefix and then cached and returned.
        """
    if not self._cached_version:
        self._cached_version = '%s%s' % (prefix, self.version_string())
    return self._cached_version