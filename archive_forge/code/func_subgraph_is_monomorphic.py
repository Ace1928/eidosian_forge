import sys
def subgraph_is_monomorphic(self):
    """Returns True if a subgraph of G1 is monomorphic to G2."""
    try:
        x = next(self.subgraph_monomorphisms_iter())
        return True
    except StopIteration:
        return False