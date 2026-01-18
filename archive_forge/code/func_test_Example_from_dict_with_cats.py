import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['This', 'is', 'a', 'sentence'], 'cats': {'cat1': 1.0, 'cat2': 0.0, 'cat3': 0.5}}])
def test_Example_from_dict_with_cats(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.cats)) == 3
    assert example.reference.cats['cat1'] == 1.0
    assert example.reference.cats['cat2'] == 0.0
    assert example.reference.cats['cat3'] == 0.5