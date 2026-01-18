from spacy.compat import pickle
from spacy.language import Language
def test_hooks_unpickle():

    def inner_func(d1, d2):
        return 'hello!'
    nlp = Language()
    doc = nlp('Hello')
    doc.user_hooks['similarity'] = inner_func
    b = pickle.dumps(doc)
    doc2 = pickle.loads(b)
    assert doc2.similarity(None) == 'hello!'