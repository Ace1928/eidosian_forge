import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'entities': [(7, 15, 'LOC'), (20, 26, 'LOC')]}])
def test_Example_from_dict_with_entities(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.ents)) == 2
    assert [example.reference[i].ent_iob_ for i in range(7)] == ['O', 'O', 'B', 'I', 'O', 'B', 'O']
    assert example.get_aligned('ENT_IOB') == [2, 2, 3, 1, 2, 3, 2]
    assert example.reference[2].ent_type_ == 'LOC'
    assert example.reference[3].ent_type_ == 'LOC'
    assert example.reference[5].ent_type_ == 'LOC'