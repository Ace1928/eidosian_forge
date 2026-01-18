from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
def get_prefix(self, fullname):
    """Return an object whose key is a prefix of the supplied value.

        :fullname: The name to find a prefix for
        :return: a tuple of (object, remainder), where the remainder is the
            portion of the name that did not match the key.
        """
    for key in self.keys():
        if fullname.startswith(key):
            return (self.get(key), fullname[len(key):])