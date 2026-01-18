import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def planar_drawing_conforms_to_embedding(embedding, pos):
    """Checks if pos conforms to the planar embedding

    Returns true iff the neighbors are actually oriented in the orientation
    specified of the embedding
    """
    for v in embedding:
        nbr_vectors = []
        v_pos = pos[v]
        for nbr in embedding[v]:
            new_vector = Vector(pos[nbr][0] - v_pos[0], pos[nbr][1] - v_pos[1], nbr)
            nbr_vectors.append(new_vector)
        nbr_vectors.sort()
        for idx, nbr_vector in enumerate(nbr_vectors):
            cw_vector = nbr_vectors[(idx + 1) % len(nbr_vectors)]
            ccw_vector = nbr_vectors[idx - 1]
            if embedding[v][nbr_vector.node]['cw'] != cw_vector.node or embedding[v][nbr_vector.node]['ccw'] != ccw_vector.node:
                return False
            if cw_vector.node != nbr_vector.node and cw_vector == nbr_vector:
                return False
            if ccw_vector.node != nbr_vector.node and ccw_vector == nbr_vector:
                return False
    return True