import copy, logging
from pyomo.common.dependencies import numpy
def all_cycles(self, G):
    """
        This function finds all the cycles in a directed graph.
        The algorithm is based on Tarjan 1973 Enumeration of the
        elementary circuits of a directed graph, SIAM J. Comput. v3 n2 1973.

        Returns
        -------
            cycleNodes
                List of lists of nodes in each cycle
            cycleEdges
                List of lists of edge indexes in each cycle
        """

    def backtrack(v, pre_key=None):
        f = False
        pointStack.append((v, pre_key))
        mark[v] = True
        markStack.append(v)
        sucs = list(adj[v])
        for si, key in sucs:
            if si < ni:
                adj[v].remove((si, key))
            elif si == ni:
                f = True
                cyc = list(pointStack)
                cyc.append((si, key))
                cycles.append(cyc)
            elif not mark[si]:
                g = backtrack(si, key)
                f = f or g
        if f:
            while markStack[-1] != v:
                u = markStack.pop()
                mark[u] = False
            markStack.pop()
            mark[v] = False
        pointStack.pop()
        return f
    i2n, adj, _ = self.adj_lists(G, multi=True)
    pointStack = []
    markStack = []
    cycles = []
    mark = [False] * len(i2n)
    for ni in range(len(i2n)):
        backtrack(ni)
        while len(markStack) > 0:
            i = markStack.pop()
            mark[i] = False
    cycleNodes = []
    for cycle in cycles:
        cycleNodes.append([])
        for i in range(len(cycle)):
            ni, key = cycle[i]
            cycle[i] = (i2n[ni], key)
            cycleNodes[-1].append(i2n[ni])
        cycleNodes[-1].pop()
    edge_map = self.edge_to_idx(G)
    cycleEdges = []
    for cyc in cycles:
        ecyc = []
        for i in range(len(cyc) - 1):
            pre, suc, key = (cyc[i][0], cyc[i + 1][0], cyc[i + 1][1])
            ecyc.append(edge_map[pre, suc, key])
        cycleEdges.append(ecyc)
    return (cycleNodes, cycleEdges)