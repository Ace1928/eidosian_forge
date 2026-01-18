import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_merge_children(en_tokenizer):
    """Test that attachments work correctly after merging."""
    text = 'WKRO played songs by the beach boys all night'
    attrs = {'tag': 'NAMED', 'lemma': 'LEMMA', 'ent_type': 'TYPE'}
    doc = en_tokenizer(text)
    assert len(doc) == 9
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[4:7], attrs=attrs)
    for word in doc:
        if word.i < word.head.i:
            assert word in list(word.head.lefts)
        elif word.i > word.head.i:
            assert word in list(word.head.rights)