import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['This', 'is', 'one', 'sentence', 'this', 'is', 'another'], 'sent_starts': [1, False, 0, None, True, -1, -5.7]}])
def test_Example_from_dict_with_sent_start(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.sents)) == 2
    for i, token in enumerate(example.reference):
        if to_ternary_int(annots['sent_starts'][i]) == 1:
            assert token.is_sent_start is True
        elif to_ternary_int(annots['sent_starts'][i]) == 0:
            assert token.is_sent_start is None
        else:
            assert token.is_sent_start is False