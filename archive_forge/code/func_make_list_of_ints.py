import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def make_list_of_ints(sequence):
    """Return list of ints from sequence of integral numbers.

    All elements of the sequence must satisfy int(element) == element
    or a ValueError is raised. Sequence is iterated through once.

    If sequence is a list, the non-int values are replaced with ints.
    So, no new list is created
    """
    if not isinstance(sequence, list):
        result = []
        for i in sequence:
            errmsg = f'sequence is not all integers: {i}'
            try:
                ii = int(i)
            except ValueError:
                raise nx.NetworkXError(errmsg) from None
            if ii != i:
                raise nx.NetworkXError(errmsg)
            result.append(ii)
        return result
    for indx, i in enumerate(sequence):
        errmsg = f'sequence is not all integers: {i}'
        if isinstance(i, int):
            continue
        try:
            ii = int(i)
        except ValueError:
            raise nx.NetworkXError(errmsg) from None
        if ii != i:
            raise nx.NetworkXError(errmsg)
        sequence[indx] = ii
    return sequence