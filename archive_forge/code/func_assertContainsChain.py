from itertools import cycle, islice
import pytest
import networkx as nx
def assertContainsChain(self, chain, expected):
    reversed_chain = list(reversed([tuple(reversed(e)) for e in chain]))
    for candidate in expected:
        if cyclic_equals(chain, candidate):
            break
        if cyclic_equals(reversed_chain, candidate):
            break
    else:
        self.fail('chain not found')