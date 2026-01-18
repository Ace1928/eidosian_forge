from collections import defaultdict
import networkx as nx
def make_bi_connected(embedding, starting_node, outgoing_node, edges_counted):
    """Triangulate a face and make it 2-connected

    This method also adds all edges on the face to `edges_counted`.

    Parameters
    ----------
    embedding: nx.PlanarEmbedding
        The embedding that defines the faces
    starting_node : node
        A node on the face
    outgoing_node : node
        A node such that the half edge (starting_node, outgoing_node) belongs
        to the face
    edges_counted: set
        Set of all half-edges that belong to a face that have been visited

    Returns
    -------
    face_nodes: list
        A list of all nodes at the border of this face
    """
    if (starting_node, outgoing_node) in edges_counted:
        return []
    edges_counted.add((starting_node, outgoing_node))
    v1 = starting_node
    v2 = outgoing_node
    face_list = [starting_node]
    face_set = set(face_list)
    _, v3 = embedding.next_face_half_edge(v1, v2)
    while v2 != starting_node or v3 != outgoing_node:
        if v1 == v2:
            raise nx.NetworkXException('Invalid half-edge')
        if v2 in face_set:
            embedding.add_half_edge_cw(v1, v3, v2)
            embedding.add_half_edge_ccw(v3, v1, v2)
            edges_counted.add((v2, v3))
            edges_counted.add((v3, v1))
            v2 = v1
        else:
            face_set.add(v2)
            face_list.append(v2)
        v1 = v2
        v2, v3 = embedding.next_face_half_edge(v2, v3)
        edges_counted.add((v1, v2))
    return face_list