from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def shift_loop_to_start_with_edge_loop(self):
    for i, edge in enumerate(self.loop):
        if edge.subcomplex_type == 'edgeLoop':
            self.loop = self.loop[i:] + self.loop[:i]
            return
    raise VertexHasNoApproxEdgeError('Vertex has no approx edge')