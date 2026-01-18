from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def appendPathNode(self, name):
    if not self.tail:
        return
    pathnode = PathNode(name)
    self.graph.connect(self.tail, pathnode)
    self.tail = pathnode
    return pathnode