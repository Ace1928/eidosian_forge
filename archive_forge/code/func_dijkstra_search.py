import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def dijkstra_search(graph, source, weight_fn, visitor):
    """Dijkstra traversal of a graph.

    The pseudo-code for the Dijkstra algorithm is listed below, with the annotated
    event points, for which the given visitor object will be called with the
    appropriate method.

    ::

        DIJKSTRA(G, source, weight)
          for each vertex u in V
              d[u] := infinity
              p[u] := u
          end for
          d[source] := 0
          INSERT(Q, source)
          while (Q != Ø)
              u := EXTRACT-MIN(Q)                         discover vertex u
              for each vertex v in Adj[u]                 examine edge (u,v)
                  if (weight[(u,v)] + d[u] < d[v])        edge (u,v) relaxed
                      d[v] := weight[(u,v)] + d[u]
                      p[v] := u
                      DECREASE-KEY(Q, v)
                  else                                    edge (u,v) not relaxed
                      ...
                  if (d[v] was originally infinity)
                      INSERT(Q, v)
              end for                                     finish vertex u
          end while

    If an exception is raised inside the callback function, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
    will return but without raising back the exception. You can also prune part of the
    search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    .. note::

        Graph can **not** be mutated while traversing.

    :param graph: The graph to be used. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`.
    :param List[int] source: An optional list of node indices to use as the starting nodes
        for the dijkstra search. If this is not specified then a source
        will be chosen arbitrarly and repeated until all components of the
        graph are searched.
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float which
        will be used to represent the weight/cost of the edge. If not specified,
        a default value of cost ``1.0`` will be used for each edge.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.DijkstraVisitor`.
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))