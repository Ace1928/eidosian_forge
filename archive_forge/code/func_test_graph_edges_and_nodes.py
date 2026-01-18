from spacy.tokens.doc import Doc
from spacy.tokens.graph import Graph
from spacy.vocab import Vocab
def test_graph_edges_and_nodes():
    doc = Doc(Vocab(), words=['a', 'b', 'c', 'd'])
    graph = Graph(doc, name='hello')
    node1 = graph.add_node((0,))
    assert graph.get_node((0,)) == node1
    node2 = graph.add_node((1, 3))
    assert list(node2) == [1, 3]
    graph.add_edge(node1, node2, label='one', weight=-10.5)
    assert graph.has_edge(node1, node2, label='one')
    assert node1.heads() == []
    assert [tuple(h) for h in node2.heads()] == [(0,)]
    assert [tuple(t) for t in node1.tails()] == [(1, 3)]
    assert [tuple(t) for t in node2.tails()] == []