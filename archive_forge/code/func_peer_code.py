import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def peer_code(self):
    peer = dict((c.peer_info() for c in self.crossings))
    even_labels = enumerate_lists(self.link_components, n=0, filter=lambda x: x % 2 == 0)
    ans = '[' + ','.join((repr([peer[c][0] for c in comp])[1:-1].replace(',', '') for comp in even_labels))
    table = ['_', '+', '-']
    ans += '] / ' + ' '.join((table[peer[c][1]] for c in sum(even_labels, [])))
    return ans