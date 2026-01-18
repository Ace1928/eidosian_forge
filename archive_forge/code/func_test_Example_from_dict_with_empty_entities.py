import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_from_dict_with_empty_entities():
    annots = {'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'entities': []}
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert example.reference.has_annotation('ENT_IOB')
    assert len(list(example.reference.ents)) == 0
    assert all((token.ent_iob_ == 'O' for token in example.reference))
    annots['entities'] = None
    example = Example.from_dict(predicted, annots)
    assert not example.reference.has_annotation('ENT_IOB')
    annots.pop('entities', None)
    example = Example.from_dict(predicted, annots)
    assert not example.reference.has_annotation('ENT_IOB')