import pickle
import hypothesis.strategies as st
import pytest
from hypothesis import given
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline._edit_tree_internals.edit_trees import EditTrees
from spacy.strings import StringStore
from spacy.training import Example
from spacy.util import make_tempdir
@pytest.mark.parametrize('top_k', (1, 5, 30))
def test_no_data(top_k):
    TEXTCAT_DATA = [("I'm so happy.", {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}), ("I'm so angry", {'cats': {'POSITIVE': 0.0, 'NEGATIVE': 1.0}})]
    nlp = English()
    nlp.add_pipe('trainable_lemmatizer', config={'top_k': top_k})
    nlp.add_pipe('textcat')
    train_examples = []
    for t in TEXTCAT_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    with pytest.raises(ValueError):
        nlp.initialize(get_examples=lambda: train_examples)