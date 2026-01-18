from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
@property
def incoming_edges(self):
    return get_incoming_edges(self, self.incoming_edge_map)