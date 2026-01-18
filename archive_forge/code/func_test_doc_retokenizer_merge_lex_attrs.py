import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenizer_merge_lex_attrs(en_vocab):
    """Test that retokenization also sets attributes on the lexeme if they're
    lexical attributes. For example, if a user sets IS_STOP, it should mean that
    "all tokens with that lexeme" are marked as a stop word, so the ambiguity
    here is acceptable. Also see #2390.
    """
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    assert not any((t.is_stop for t in doc))
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2], attrs={'lemma': 'hello world', 'is_stop': True})
    assert doc[0].lemma_ == 'hello world'
    assert doc[0].is_stop
    doc = Doc(en_vocab, words=['eins', 'zwei', '!', '!'])
    assert not any((t.like_num for t in doc))
    assert not any((t.is_stop for t in doc))
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2], attrs={'like_num': True})
        retokenizer.merge(doc[2:4], attrs={'is_stop': True})
    assert doc[0].like_num
    assert doc[1].is_stop
    assert not doc[0].is_stop
    assert not doc[1].like_num
    doc = Doc(en_vocab, words=['eins', 'zwei', '!', '!'])
    assert doc[0].norm_ == 'eins'
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:1], attrs={'norm': '1'})
    assert doc[0].norm_ == '1'
    assert en_vocab['eins'].norm_ == 'eins'