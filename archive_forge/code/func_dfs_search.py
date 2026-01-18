import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def dfs_search(graph, source, visitor):
    """Depth-first traversal of a directed/undirected graph.

    The pseudo-code for the DFS algorithm is listed below, with the annotated
    event points, for which the given visitor object will be called with the
    appropriate method.

    ::

        DFS(G)
          for each vertex u in V
              color[u] := WHITE                 initialize vertex u
          end for
          time := 0
          call DFS-VISIT(G, source)             start vertex s

        DFS-VISIT(G, u)
          color[u] := GRAY                      discover vertex u
          for each v in Adj[u]                  examine edge (u,v)
              if (color[v] = WHITE)             (u,v) is a tree edge
                  all DFS-VISIT(G, v)
              else if (color[v] = GRAY)         (u,v) is a back edge
              ...
             else if (color[v] = BLACK)         (u,v) is a cross or forward edge
             ...
          end for
          color[u] := BLACK                     finish vertex u

    If an exception is raised inside the callback function, the graph traversal
    will be stopped immediately. You can exploit this to exit early by raising a
    :class:`~rustworkx.visit.StopSearch` exception. You can also prune part of the
    search tree by raising :class:`~rustworkx.visit.PruneSearch`.

    In the following example we keep track of the tree edges:

    .. jupyter-execute::

           import rustworkx as rx
           from rustworkx.visit import DFSVisitor

           class TreeEdgesRecorder(DFSVisitor):

               def __init__(self):
                   self.edges = []

               def tree_edge(self, edge):
                   self.edges.append(edge)

           graph = rx.PyGraph()
           graph.extend_from_edge_list([(1, 3), (0, 1), (2, 1), (0, 2)])
           vis = TreeEdgesRecorder()
           rx.dfs_search(graph, [0], vis)
           print('Tree edges:', vis.edges)

    .. note::

        Graph can *not* be mutated while traversing.

    :param PyGraph graph: The graph to be used.
    :param List[int] source: An optional list of node indices to use as the starting
        nodes for the depth-first search. If this is not specified then a source
        will be chosen arbitrarly and repeated until all components of the
        graph are searched.
    :param visitor: A visitor object that is invoked at the event points inside the
        algorithm. This should be a subclass of :class:`~rustworkx.visit.DFSVisitor`.
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))