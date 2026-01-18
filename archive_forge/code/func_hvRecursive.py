import warnings
import numpy
def hvRecursive(self, dimIndex, length, bounds):
    """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
    hvol = 0.0
    sentinel = self.list.sentinel
    if length == 0:
        return hvol
    elif dimIndex == 0:
        return -sentinel.next[0].cargo[0]
    elif dimIndex == 1:
        q = sentinel.next[1]
        h = q.cargo[0]
        p = q.next[1]
        while p is not sentinel:
            pCargo = p.cargo
            hvol += h * (q.cargo[1] - pCargo[1])
            if pCargo[0] < h:
                h = pCargo[0]
            q = p
            p = q.next[1]
        hvol += h * q.cargo[1]
        return hvol
    else:
        remove = self.list.remove
        reinsert = self.list.reinsert
        hvRecursive = self.hvRecursive
        p = sentinel
        q = p.prev[dimIndex]
        while q.cargo != None:
            if q.ignore < dimIndex:
                q.ignore = 0
            q = q.prev[dimIndex]
        q = p.prev[dimIndex]
        while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
            p = q
            remove(p, dimIndex, bounds)
            q = p.prev[dimIndex]
            length -= 1
        qArea = q.area
        qCargo = q.cargo
        qPrevDimIndex = q.prev[dimIndex]
        if length > 1:
            hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
        else:
            qArea[0] = 1
            qArea[1:dimIndex + 1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
        q.volume[dimIndex] = hvol
        if q.ignore >= dimIndex:
            qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
        else:
            qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
            if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                q.ignore = dimIndex
        while p is not sentinel:
            pCargoDimIndex = p.cargo[dimIndex]
            hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
            bounds[dimIndex] = pCargoDimIndex
            reinsert(p, dimIndex, bounds)
            length += 1
            q = p
            p = p.next[dimIndex]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
            else:
                q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                    q.ignore = dimIndex
        hvol -= q.area[dimIndex] * q.cargo[dimIndex]
        return hvol