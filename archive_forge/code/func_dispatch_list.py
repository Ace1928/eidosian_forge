from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def dispatch_list(self, node_list):
    for node in node_list:
        self.dispatch(node)