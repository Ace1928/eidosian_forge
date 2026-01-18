from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def minima(self, size, ordering=[]):
    """
        Return the minimal complexity of all rotations and inverted
        rotations, and a list of the words and orderings that realize
        the minimal complexity.
        """
    least = Complexity([])
    minima = []
    for word in (self, ~self):
        for n in range(len(self)):
            complexity, Xordering = word.complexity(size, ordering, spin=n)
            if complexity < least:
                least = complexity
                minima = [(CyclicWord(word.spun(n)), Xordering)]
            elif complexity == least:
                minima.append((CyclicWord(word.spun(n)), Xordering))
    return (least, minima)