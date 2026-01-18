from itertools import chain
from nltk.internals import Counter
def safeappend(self, key, item):
    """
        Append 'item' to the list at 'key'.  If no list exists for 'key', then
        construct one.
        """
    if key not in self:
        self[key] = []
    self[key].append(item)