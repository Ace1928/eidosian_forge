from spacy.tokens.doc import Doc
from spacy.tokens.graph import Graph
from spacy.vocab import Vocab
def test_graph_init():
    doc = Doc(Vocab(), words=['a', 'b', 'c', 'd'])
    graph = Graph(doc, name='hello')
    assert graph.name == 'hello'
    assert graph.doc is doc