import networkx as nx
from collections import deque
def cut_pairs(self):
    """
        Return a list of cut_pairs.  The graph is assumed to be
        connected and to have no cut vertices.
        """
    pairs = []
    majors = [v for v in self.vertices if self.valence(v) > 2]
    if len(majors) == 2:
        v, V = majors
        if self.valence(v) == 3:
            return []
        edge = self.find_edge[v, V]
        if not edge or edge.multiplicity < 2:
            return []
        return majors
    major_set = set(majors)
    for n in range(1, len(majors)):
        for v in majors[:n]:
            pair = (v, majors[n])
            components = self.components(deleted_vertices=pair)
            if len(components) > 2:
                pairs.append(pair)
            elif len(components) == 2:
                M0 = len(major_set & components[0])
                M1 = len(major_set & components[1])
                edge = self.find_edge[pair]
                if edge:
                    if M0 or M1:
                        pairs.append(pair)
                        continue
                elif M0 and M1:
                    pairs.append(pair)
    return pairs