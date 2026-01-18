import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'entities': [(7, 15, 'LOC'), (11, 15, 'LOC'), (20, 26, 'LOC')]}])
def test_Example_from_dict_with_entities_overlapping(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    with pytest.raises(ValueError):
        Example.from_dict(predicted, annots)