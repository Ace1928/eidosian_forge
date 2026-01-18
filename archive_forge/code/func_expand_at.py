from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def expand_at(self, position):
    edge = self.loop[position]
    if edge.subcomplex_type != 'beta':
        return position
    other_tet_index, glued_perm = self.truncated_complex.get_glued_tet_and_perm(edge.tet_and_perm)
    other_perm = glued_perm * t3m.Perm4([0, 2, 1, 3])
    if (other_tet_index, other_perm.tuple()) in self.used_hexagons:
        return position
    hex = self.truncated_complex.get_edges_of_small_hexagon((other_tet_index, other_perm))
    self.mark_hexagon_as_used(hex)
    for i, new_edge in enumerate(hex[1:]):
        if i == 0:
            self.loop[position] = new_edge
        else:
            position += 1
            self.loop.insert(position, new_edge)
            position = self.insert_edge_loop(position)
    return position