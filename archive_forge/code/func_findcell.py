import re
from collections import defaultdict
from operator import itemgetter
from nltk.tree.tree import Tree
from nltk.util import OrderedDict
def findcell(m, matrix, startoflevel, children):
    """
            Find vacant row, column index for node ``m``.
            Iterate over current rows for this level (try lowest first)
            and look for cell between first and last child of this node,
            add new row to level if no free row available.
            """
    candidates = [a for _, a in children[m]]
    minidx, maxidx = (min(candidates), max(candidates))
    leaves = tree[m].leaves()
    center = scale * sum(leaves) // len(leaves)
    if minidx < maxidx and (not minidx < center < maxidx):
        center = sum(candidates) // len(candidates)
    if max(candidates) - min(candidates) > 2 * scale:
        center -= center % scale
        if minidx < maxidx and (not minidx < center < maxidx):
            center += scale
    if ids[m] == 0:
        startoflevel = len(matrix)
    for rowidx in range(startoflevel, len(matrix) + 1):
        if rowidx == len(matrix):
            matrix.append([vertline if a not in (corner, None) else None for a in matrix[-1]])
        row = matrix[rowidx]
        if len(children[m]) == 1:
            return (rowidx, next(iter(children[m]))[1])
        elif all((a is None or a == vertline for a in row[min(candidates):max(candidates) + 1])):
            for n in range(scale):
                i = j = center + n
                while j > minidx or i < maxidx:
                    if i < maxidx and (matrix[rowidx][i] is None or i in candidates):
                        return (rowidx, i)
                    elif j > minidx and (matrix[rowidx][j] is None or j in candidates):
                        return (rowidx, j)
                    i += scale
                    j -= scale
    raise ValueError('could not find a free cell for:\n%s\n%smin=%d; max=%d' % (tree[m], minidx, maxidx, dumpmatrix()))