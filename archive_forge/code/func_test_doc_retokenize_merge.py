import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_merge(en_tokenizer):
    text = 'WKRO played songs by the beach boys all night'
    attrs = {'tag': 'NAMED', 'lemma': 'LEMMA', 'ent_type': 'TYPE', 'morph': 'Number=Plur'}
    doc = en_tokenizer(text)
    assert len(doc) == 9
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[4:7], attrs=attrs)
        retokenizer.merge(doc[7:9], attrs=attrs)
    assert len(doc) == 6
    assert doc[4].text == 'the beach boys'
    assert doc[4].text_with_ws == 'the beach boys '
    assert doc[4].tag_ == 'NAMED'
    assert doc[4].lemma_ == 'LEMMA'
    assert str(doc[4].morph) == 'Number=Plur'
    assert doc[5].text == 'all night'
    assert doc[5].text_with_ws == 'all night'
    assert doc[5].tag_ == 'NAMED'
    assert str(doc[5].morph) == 'Number=Plur'
    assert doc[5].lemma_ == 'LEMMA'