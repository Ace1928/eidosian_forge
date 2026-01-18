import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
def parseData():
    """Returns stream of pairs b[i], x[i] for sparse6 format."""
    chunks = iter(data)
    d = None
    dLen = 0
    while 1:
        if dLen < 1:
            try:
                d = next(chunks)
            except StopIteration:
                return
            dLen = 6
        dLen -= 1
        b = d >> dLen & 1
        x = d & (1 << dLen) - 1
        xLen = dLen
        while xLen < k:
            try:
                d = next(chunks)
            except StopIteration:
                return
            dLen = 6
            x = (x << 6) + d
            xLen += 6
        x = x >> xLen - k
        dLen = xLen - k
        yield (b, x)