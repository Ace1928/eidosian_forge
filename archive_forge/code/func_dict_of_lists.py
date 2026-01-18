import cgi
import copy
import sys
from collections.abc import MutableMapping as DictMixin
def dict_of_lists(self):
    """
        Returns a dictionary where each key is associated with a
        list of values.
        """
    unicode_dict = {}
    for key, value in self.multi.dict_of_lists().items():
        value = [self._decode_value(value) for value in value]
        unicode_dict[self._decode_key(key)] = value
    return unicode_dict