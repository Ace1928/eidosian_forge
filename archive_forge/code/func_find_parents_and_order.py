from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.basePen import AbstractPen, BasePen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from math import sqrt, copysign, atan2, pi
from enum import Enum
import itertools
import logging
def find_parents_and_order(glyphsets, locations):
    parents = [None] + list(range(len(glyphsets) - 1))
    order = list(range(len(glyphsets)))
    if locations:
        bases = (i for i, l in enumerate(locations) if all((v == 0 for v in l.values())))
        if bases:
            base = next(bases)
            logging.info('Base master index %s, location %s', base, locations[base])
        else:
            base = 0
            logging.warning('No base master location found')
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            graph = [[0] * len(locations) for _ in range(len(locations))]
            axes = set()
            for l in locations:
                axes.update(l.keys())
            axes = sorted(axes)
            vectors = [tuple((l.get(k, 0) for k in axes)) for l in locations]
            for i, j in itertools.combinations(range(len(locations)), 2):
                graph[i][j] = vdiff_hypot2(vectors[i], vectors[j])
            tree = minimum_spanning_tree(graph)
            rows, cols = tree.nonzero()
            graph = defaultdict(set)
            for row, col in zip(rows, cols):
                graph[row].add(col)
                graph[col].add(row)
            parents = [None] * len(locations)
            order = []
            visited = set()
            queue = deque([base])
            while queue:
                i = queue.popleft()
                visited.add(i)
                order.append(i)
                for j in sorted(graph[i]):
                    if j not in visited:
                        parents[j] = i
                        queue.append(j)
        except ImportError:
            pass
        log.info('Parents: %s', parents)
        log.info('Order: %s', order)
    return (parents, order)