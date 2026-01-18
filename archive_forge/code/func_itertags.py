import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def itertags(self):
    """
        iterate on tags.

        :rtype: iterator
        """
    for tag in self.__tags:
        yield tag