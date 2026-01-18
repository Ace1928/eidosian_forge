import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
@pytest.mark.issue(3012)
def test_issue3012(en_vocab):
    """Test that the is_tagged attribute doesn't get overwritten when we from_array
    without tag information."""
    words = ['This', 'is', '10', '%', '.']
    tags = ['DT', 'VBZ', 'CD', 'NN', '.']
    pos = ['DET', 'VERB', 'NUM', 'NOUN', 'PUNCT']
    ents = ['O', 'O', 'B-PERCENT', 'I-PERCENT', 'O']
    doc = Doc(en_vocab, words=words, tags=tags, pos=pos, ents=ents)
    assert doc.has_annotation('TAG')
    expected = ('10', 'NUM', 'CD', 'PERCENT')
    assert (doc[2].text, doc[2].pos_, doc[2].tag_, doc[2].ent_type_) == expected
    header = [ENT_IOB, ENT_TYPE]
    ent_array = doc.to_array(header)
    doc.from_array(header, ent_array)
    assert (doc[2].text, doc[2].pos_, doc[2].tag_, doc[2].ent_type_) == expected
    doc_bytes = doc.to_bytes()
    doc2 = Doc(en_vocab).from_bytes(doc_bytes)
    assert (doc2[2].text, doc2[2].pos_, doc2[2].tag_, doc2[2].ent_type_) == expected