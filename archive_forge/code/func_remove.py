import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def remove(self, value):
    """
        Remove a given value from the list.

        :param value: object

        """
    found = False
    for i in range(len(self)):
        if self[i] == value:
            found = True
            break
    if found:
        self.pop(i)