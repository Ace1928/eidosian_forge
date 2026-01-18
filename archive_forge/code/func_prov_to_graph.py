import networkx as nx
from prov.model import (
def prov_to_graph(prov_document):
    """
    Convert a :class:`~prov.model.ProvDocument` to a `MultiDiGraph
    <https://networkx.github.io/documentation/stable/reference/classes/multidigraph.html>`_
    instance of the `NetworkX <https://networkx.github.io/>`_ library.

    :param prov_document: The :class:`~prov.model.ProvDocument` instance to convert.
    """
    g = nx.MultiDiGraph()
    unified = prov_document.unified()
    node_map = dict()
    for element in unified.get_records(ProvElement):
        g.add_node(element)
        node_map[element.identifier] = element
    for relation in unified.get_records(ProvRelation):
        attr_pair_1, attr_pair_2 = relation.formal_attributes[:2]
        qn1, qn2 = (attr_pair_1[1], attr_pair_2[1])
        if qn1 and qn2:
            try:
                if qn1 not in node_map:
                    node_map[qn1] = INFERRED_ELEMENT_CLASS[attr_pair_1[0]](None, qn1)
                if qn2 not in node_map:
                    node_map[qn2] = INFERRED_ELEMENT_CLASS[attr_pair_2[0]](None, qn2)
            except KeyError:
                continue
            g.add_edge(node_map[qn1], node_map[qn2], relation=relation)
    return g