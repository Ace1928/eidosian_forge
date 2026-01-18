import re
from collections import defaultdict
from operator import itemgetter
from nltk.tree.tree import Tree
from nltk.util import OrderedDict
@staticmethod
def nodecoords(tree, sentence, highlight):
    """
        Produce coordinates of nodes on a grid.

        Objective:

        - Produce coordinates for a non-overlapping placement of nodes and
            horizontal lines.
        - Order edges so that crossing edges cross a minimal number of previous
            horizontal lines (never vertical lines).

        Approach:

        - bottom up level order traversal (start at terminals)
        - at each level, identify nodes which cannot be on the same row
        - identify nodes which cannot be in the same column
        - place nodes into a grid at (row, column)
        - order child-parent edges with crossing edges last

        Coordinates are (row, column); the origin (0, 0) is at the top left;
        the root node is on row 0. Coordinates do not consider the size of a
        node (which depends on font, &c), so the width of a column of the grid
        should be automatically determined by the element with the greatest
        width in that column. Alternatively, the integer coordinates could be
        converted to coordinates in which the distances between adjacent nodes
        are non-uniform.

        Produces tuple (nodes, coords, edges, highlighted) where:

        - nodes[id]: Tree object for the node with this integer id
        - coords[id]: (n, m) coordinate where to draw node with id in the grid
        - edges[id]: parent id of node with this id (ordered dictionary)
        - highlighted: set of ids that should be highlighted
        """

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

    def dumpmatrix():
        """Dump matrix contents for debugging purposes."""
        return '\n'.join(('%2d: %s' % (n, ' '.join((('%2r' % i)[:2] for i in row))) for n, row in enumerate(matrix)))
    leaves = tree.leaves()
    if not all((isinstance(n, int) for n in leaves)):
        raise ValueError('All leaves must be integer indices.')
    if len(leaves) != len(set(leaves)):
        raise ValueError('Indices must occur at most once.')
    if not all((0 <= n < len(sentence) for n in leaves)):
        raise ValueError('All leaves must be in the interval 0..n with n=len(sentence)\ntokens: %d indices: %r\nsentence: %s' % (len(sentence), tree.leaves(), sentence))
    vertline, corner = (-1, -2)
    tree = tree.copy(True)
    for a in tree.subtrees():
        a.sort(key=lambda n: min(n.leaves()) if isinstance(n, Tree) else n)
    scale = 2
    crossed = set()
    positions = tree.treepositions()
    maxdepth = max(map(len, positions)) + 1
    childcols = defaultdict(set)
    matrix = [[None] * (len(sentence) * scale)]
    nodes = {}
    ids = {a: n for n, a in enumerate(positions)}
    highlighted_nodes = {n for a, n in ids.items() if not highlight or tree[a] in highlight}
    levels = {n: [] for n in range(maxdepth - 1)}
    terminals = []
    for a in positions:
        node = tree[a]
        if isinstance(node, Tree):
            levels[maxdepth - node.height()].append(a)
        else:
            terminals.append(a)
    for n in levels:
        levels[n].sort(key=lambda n: max(tree[n].leaves()) - min(tree[n].leaves()))
    terminals.sort()
    positions = set(positions)
    for m in terminals:
        i = int(tree[m]) * scale
        assert matrix[0][i] is None, (matrix[0][i], m, i)
        matrix[0][i] = ids[m]
        nodes[ids[m]] = sentence[tree[m]]
        if nodes[ids[m]] is None:
            nodes[ids[m]] = '...'
            highlighted_nodes.discard(ids[m])
        positions.remove(m)
        childcols[m[:-1]].add((0, i))
    for n in sorted(levels, reverse=True):
        nodesatdepth = levels[n]
        startoflevel = len(matrix)
        matrix.append([vertline if a not in (corner, None) else None for a in matrix[-1]])
        for m in nodesatdepth:
            if n < maxdepth - 1 and childcols[m]:
                _, pivot = min(childcols[m], key=itemgetter(1))
                if {a[:-1] for row in matrix[:-1] for a in row[:pivot] if isinstance(a, tuple)} & {a[:-1] for row in matrix[:-1] for a in row[pivot:] if isinstance(a, tuple)}:
                    crossed.add(m)
            rowidx, i = findcell(m, matrix, startoflevel, childcols)
            positions.remove(m)
            for _, x in childcols[m]:
                matrix[rowidx][x] = corner
            matrix[rowidx][i] = ids[m]
            nodes[ids[m]] = tree[m]
            if len(m) > 0:
                childcols[m[:-1]].add((rowidx, i))
    assert len(positions) == 0
    for m in range(scale * len(sentence) - 1, -1, -1):
        if not any((isinstance(row[m], (Tree, int)) for row in matrix)):
            for row in matrix:
                del row[m]
    matrix = [row for row in reversed(matrix) if not all((a is None or a == vertline for a in row))]
    coords = {}
    for n, _ in enumerate(matrix):
        for m, i in enumerate(matrix[n]):
            if isinstance(i, int) and i >= 0:
                coords[i] = (n, m)
    positions = sorted((a for level in levels.values() for a in level), key=lambda a: a[:-1] in crossed)
    edges = OrderedDict()
    for i in reversed(positions):
        for j, _ in enumerate(tree[i]):
            edges[ids[i + (j,)]] = ids[i]
    return (nodes, coords, edges, highlighted_nodes)