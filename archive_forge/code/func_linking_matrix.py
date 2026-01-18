from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def linking_matrix(self):
    """
        Calculates the linking number for each pair of link components.

        Returns a linking matrix, in which the (i,j)th component is the
        linking number of the ith and jth link components.
        """
    mat = [[0 for i in range(len(self.link_components))] for j in range(len(self.link_components))]
    for n1, comp1 in enumerate(self.link_components):
        for n2, comp2 in enumerate(self.link_components):
            tally = [[0 for m in range(len(self.crossings))] for n in range(2)]
            if comp1 != comp2:
                for i, c in enumerate(self.crossings):
                    for x1 in comp1:
                        if x1[0] == c:
                            tally[0][i] += 1
                    for x2 in comp2:
                        if x2[0] == c:
                            tally[1][i] += 1
            for k, c in enumerate(self.crossings):
                if tally[0][k] == 1 and tally[1][k] == 1:
                    mat[n1][n2] += 0.5 * c.sign
            mat[n1][n2] = int(mat[n1][n2])
    return mat