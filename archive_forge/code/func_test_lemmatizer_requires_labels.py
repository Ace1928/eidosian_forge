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
def test_lemmatizer_requires_labels():
    nlp = English()
    nlp.add_pipe('trainable_lemmatizer')
    with pytest.raises(ValueError):
        nlp.initialize()