from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def eigenvalues(creation_sequence):
    """
    Return sequence of eigenvalues of the Laplacian of the threshold
    graph for the given creation_sequence.

    Based on the Ferrer's diagram method.  The spectrum is integral
    and is the conjugate of the degree sequence.

    See::

      @Article{degree-merris-1994,
       author = {Russel Merris},
       title = {Degree maximal graphs are Laplacian integral},
       journal = {Linear Algebra Appl.},
       year = {1994},
       volume = {199},
       pages = {381--389},
      }

    """
    degseq = degree_sequence(creation_sequence)
    degseq.sort()
    eiglist = []
    eig = 0
    row = len(degseq)
    bigdeg = degseq.pop()
    while row:
        if bigdeg < row:
            eiglist.append(eig)
            row -= 1
        else:
            eig += 1
            if degseq:
                bigdeg = degseq.pop()
            else:
                bigdeg = 0
    return eiglist