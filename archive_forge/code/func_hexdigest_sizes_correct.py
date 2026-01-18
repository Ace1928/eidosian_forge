import pytest
import networkx as nx
from networkx.generators import directed
def hexdigest_sizes_correct(a, digest_size):
    """
    returns True if all hex digest sizes are the expected length in a node:subgraph-hashes
    dictionary. Hex digest string length == 2 * bytes digest length since each pair of hex
    digits encodes 1 byte (https://docs.python.org/3/library/hashlib.html)
    """
    hexdigest_size = digest_size * 2
    list_digest_sizes_correct = lambda l: all((len(x) == hexdigest_size for x in l))
    return all((list_digest_sizes_correct(hashes) for hashes in a.values()))