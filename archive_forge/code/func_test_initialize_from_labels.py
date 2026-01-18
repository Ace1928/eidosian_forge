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
def test_initialize_from_labels():
    nlp = Language()
    lemmatizer = nlp.add_pipe('trainable_lemmatizer')
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda: train_examples)
    nlp2 = Language()
    lemmatizer2 = nlp2.add_pipe('trainable_lemmatizer')
    lemmatizer2.initialize(get_examples=lambda: train_examples[:1], labels=lemmatizer.label_data)
    assert lemmatizer2.tree2label == {1: 0, 3: 1, 4: 2, 6: 3}
    assert lemmatizer2.label_data == {'trees': [{'orig': 'S', 'subst': 's'}, {'prefix_len': 1, 'suffix_len': 0, 'prefix_tree': 0, 'suffix_tree': 4294967295}, {'orig': 's', 'subst': ''}, {'prefix_len': 0, 'suffix_len': 1, 'prefix_tree': 4294967295, 'suffix_tree': 2}, {'prefix_len': 0, 'suffix_len': 0, 'prefix_tree': 4294967295, 'suffix_tree': 4294967295}, {'orig': 'E', 'subst': 'e'}, {'prefix_len': 1, 'suffix_len': 0, 'prefix_tree': 5, 'suffix_tree': 4294967295}], 'labels': (1, 3, 4, 6)}