import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'entities': [(7, 15, 'LOC'), (20, 26, 'LOC')], 'links': {(7, 15): {'Q60': 1.0, 'Q64': 0.0}, (20, 26): {'Q60': 0.0, 'Q64': 1.0}}}])
def test_Example_from_dict_with_links(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert example.reference[0].ent_kb_id_ == ''
    assert example.reference[1].ent_kb_id_ == ''
    assert example.reference[2].ent_kb_id_ == 'Q60'
    assert example.reference[3].ent_kb_id_ == 'Q60'
    assert example.reference[4].ent_kb_id_ == ''
    assert example.reference[5].ent_kb_id_ == 'Q64'
    assert example.reference[6].ent_kb_id_ == ''