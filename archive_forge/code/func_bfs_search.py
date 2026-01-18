import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def bfs_search(graph, source, visitor):
    """Breadth-first traversal of a directed/undirected graph.

    The pseudo-code for the BFS algorithm is listed below, with the annotated
    event points, for which the given visitor object will be called with the
    appropriate method.

    ::

        BFS(G, s)
          for each vertex u in V
              color[u] := WHITE
          end for
          color[s] := GRAY
          EQUEUE(Q, s)                             discover vertex s
          while (Q != Ø)
              u := DEQUEUE(Q)
              for each vertex v in Adj[u]          (u,v) is a tree edge
                  if (color[v] = WHITE)
                      color[v] = GRAY
                  else                             (u,v) is a non - tree edge
                      if (color[v] = GRAY)         (u,v) has a gray target
                          ...
                      else if (color[v] = BLACK)   (u,v) has a black target
                          ...
              end for
              color[u] := BLACK                    finish vertex u
          end while

    If an exception is raised inside the callback function, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception, in which case the search function
    will return but without raising back the exception. You can also prune part of
    the search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    In the following example we keep track of the tree edges:

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visit import BFSVisitor


        class TreeEdgesRecorder(BFSVisitor):

            def __init__(self):
                self.edges = []

            def tree_edge(self, edge):
                self.edges.append(edge)

        graph = rx.PyDiGraph()
        graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
        vis = TreeEdgesRecorder()
        rx.bfs_search(graph, [0], vis)
        print('Tree edges:', vis.edges)

    .. note::

        Graph can **not** be mutated while traversing.

    :param graph: The graph to be used. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`
    :param List[int] source: An optional list of node indices to use as the starting
        nodes for the breadth-first search. If this is not specified then a source
        will be chosen arbitrarly and repeated until all components of the
        graph are searched.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.BFSVisitor`.
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))