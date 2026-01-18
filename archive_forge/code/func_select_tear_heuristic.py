import copy, logging
from pyomo.common.dependencies import numpy
def select_tear_heuristic(self, G):
    """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The secondary criteria is to
        minimize the number of tears.

        This function uses a branch and bound type approach.

        Returns
        -------
            tsets
                List of lists of tear sets. All the tear sets returned
                are equally good. There are often a very large number
                of equally good tear sets.
            upperbound_loop
                The max number of times any single loop is torn
            upperbound_total
                The total number of loops

        Improvements for the future

        I think I can improve the efficiency of this, but it is good
        enough for now. Here are some ideas for improvement:

            1. Reduce the number of redundant solutions. It is possible
            to find tears sets [1,2] and [2,1]. I eliminate
            redundant solutions from the results, but they can
            occur and it reduces efficiency.

            2. Look at strongly connected components instead of whole
            graph. This would cut back on the size of graph we are
            looking at. The flowsheets are rarely one strongly
            connected component.

            3. When you add an edge to a tear set you could reduce the
            size of the problem in the branch by only looking at
            strongly connected components with that edge removed.

            4. This returns all equally good optimal tear sets. That
            may not really be necessary. For very large flowsheets,
            there could be an extremely large number of optimal tear
            edge sets.
        """

    def sear(depth, prevY):
        for i in range(len(cycleEdges[depth])):
            y = list(prevY)
            y[cycleEdges[depth][i]] = 1
            Ay = numpy.dot(A, y)
            maxAy = max(Ay)
            sumY = sum(y)
            if maxAy > upperBound[0]:
                continue
            elif maxAy == upperBound[0] and sumY > upperBound[1]:
                continue
            if min(Ay) > 0:
                if maxAy < upperBound[0]:
                    upperBound[0] = maxAy
                    upperBound[1] = sumY
                elif sumY < upperBound[1]:
                    upperBound[1] = sumY
                ySet.append([list(y), maxAy, sumY])
            else:
                for j in range(depth + 1, nr):
                    if Ay[j] == 0:
                        sear(j, y)
    tearUB = self.tear_upper_bound(G)
    A, _, cycleEdges = self.cycle_edge_matrix(G)
    nr, nc = A.shape
    if nr == 0:
        return [[[]], 0, 0]
    y_init = [False] * G.number_of_edges()
    for j in tearUB:
        y_init[j] = 1
    Ay_init = numpy.dot(A, y_init)
    upperBound = [max(Ay_init), sum(y_init)]
    y_init = [False] * G.number_of_edges()
    ySet = []
    sear(0, y_init)
    deleteSet = []
    for i in range(len(ySet)):
        if ySet[i][1] > upperBound[0]:
            deleteSet.append(i)
        elif ySet[i][1] == upperBound[0] and ySet[i][2] > upperBound[1]:
            deleteSet.append(i)
    for i in reversed(deleteSet):
        del ySet[i]
    deleteSet = []
    for i in range(len(ySet) - 1):
        if i in deleteSet:
            continue
        for j in range(i + 1, len(ySet)):
            if j in deleteSet:
                continue
            for k in range(len(y_init)):
                eq = True
                if ySet[i][0][k] != ySet[j][0][k]:
                    eq = False
                    break
            if eq == True:
                deleteSet.append(j)
    for i in reversed(sorted(deleteSet)):
        del ySet[i]
    es = []
    for y in ySet:
        edges = []
        for i in range(len(y[0])):
            if y[0][i] == 1:
                edges.append(i)
        es.append(edges)
    return (es, upperBound[0], upperBound[1])