import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'spans': {'cities': [(7, 15, 'LOC'), (11, 15, 'LOC'), (20, 26, 'LOC')], 'people': [(0, 1, 'PERSON')]}}])
def test_Example_from_dict_with_spans_overlapping(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.ents)) == 0
    assert len(list(example.reference.spans['cities'])) == 3
    assert len(list(example.reference.spans['people'])) == 1
    for span in example.reference.spans['cities']:
        assert span.label_ == 'LOC'
    for span in example.reference.spans['people']:
        assert span.label_ == 'PERSON'