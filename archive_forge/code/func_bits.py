from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
def bits():
    """Returns sequence of individual bits from 6-bit-per-value
        list of data values."""
    for d in data:
        for i in [5, 4, 3, 2, 1, 0]:
            yield (d >> i & 1)