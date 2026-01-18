import collections
from humanfriendly.compat import basestring, unicode
def coerce_key(self, key):
    """
        Coerce string keys to :class:`CaseInsensitiveKey` objects.

        :param key: The value to coerce (any type).
        :returns: If `key` is a string then a :class:`CaseInsensitiveKey`
                  object is returned, otherwise the value of `key` is
                  returned unmodified.
        """
    if isinstance(key, basestring):
        key = CaseInsensitiveKey(key)
    return key