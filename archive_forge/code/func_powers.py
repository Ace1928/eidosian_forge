from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def powers(self, start=0):
    """
        Return a list of pairs (letter, power) for the exponential
        representation of the spun word, beginning at start.
        """
    result = []
    last_letter = self[start]
    count = 0
    for letter in self.spun(start):
        if letter == last_letter:
            count += 1
        else:
            result.append((last_letter, count))
            count = 1
            last_letter = letter
    result.append((last_letter, count))
    return result