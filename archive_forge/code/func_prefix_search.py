import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def prefix_search(self, key):
    """Collect all items that are prefixes of key.

        Prefix in this case are delineated by '.' characters so
        'foo.bar.baz' is a 3 chunk sequence of 3 "prefixes" (
        "foo", "bar", and "baz").

        """
    collected = deque()
    key_parts = key.split('.')
    current = self._root
    self._get_items(current, key_parts, collected, 0)
    return collected