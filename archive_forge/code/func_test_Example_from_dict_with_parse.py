import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'London', 'and', 'Berlin', '.'], 'deps': ['nsubj', 'ROOT', 'dobj', 'cc', 'conj', 'punct'], 'heads': [1, 1, 1, 2, 2, 1]}])
def test_Example_from_dict_with_parse(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    for i, token in enumerate(example.reference):
        assert token.dep_ == annots['deps'][i]
        assert token.head.i == annots['heads'][i]