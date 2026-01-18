import copy, logging
from pyomo.common.dependencies import numpy
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