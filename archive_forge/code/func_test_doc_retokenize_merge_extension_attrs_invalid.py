import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
@pytest.mark.parametrize('underscore_attrs', [{'a': 'x'}, {'b': 'x'}, {'c': 'x'}, [1]])
def test_doc_retokenize_merge_extension_attrs_invalid(en_vocab, underscore_attrs):
    Token.set_extension('a', getter=lambda x: x, force=True)
    Token.set_extension('b', method=lambda x: x, force=True)
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    attrs = {'_': underscore_attrs}
    with pytest.raises(ValueError):
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[0:2], attrs=attrs)