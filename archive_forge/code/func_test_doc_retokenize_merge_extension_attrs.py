import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_merge_extension_attrs(en_vocab):
    Token.set_extension('a', default=False, force=True)
    Token.set_extension('b', default='nothing', force=True)
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    with doc.retokenize() as retokenizer:
        attrs = {'lemma': 'hello world', '_': {'a': True, 'b': '1'}}
        retokenizer.merge(doc[0:2], attrs=attrs)
    assert doc[0].lemma_ == 'hello world'
    assert doc[0]._.a is True
    assert doc[0]._.b == '1'
    doc = Doc(en_vocab, words=['hello', 'world', '!', '!'])
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2], attrs={'_': {'a': True, 'b': '1'}})
        retokenizer.merge(doc[2:4], attrs={'_': {'a': None, 'b': '2'}})
    assert doc[0]._.a is True
    assert doc[0]._.b == '1'
    assert doc[1]._.a is None
    assert doc[1]._.b == '2'