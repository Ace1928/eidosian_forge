import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def iterontag(self, tag):
    """
        iterate on items marked with one given tag.

        :param tag: object
        """
    i = 0
    for onetag in self.__tags:
        if tag == onetag:
            yield self[i]
        i += 1