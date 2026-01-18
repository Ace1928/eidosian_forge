import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('text', ['Give it back! He pleaded.', ' Give it back! He pleaded. '])
def test_doc_api_serialize(en_tokenizer, text):
    tokens = en_tokenizer(text)
    tokens[0].lemma_ = 'lemma'
    tokens[0].norm_ = 'norm'
    tokens.ents = [(tokens.vocab.strings['PRODUCT'], 0, 1)]
    tokens[0].ent_kb_id_ = 'ent_kb_id'
    tokens[0].ent_id_ = 'ent_id'
    new_tokens = Doc(tokens.vocab).from_bytes(tokens.to_bytes())
    assert tokens.text == new_tokens.text
    assert [t.text for t in tokens] == [t.text for t in new_tokens]
    assert [t.orth for t in tokens] == [t.orth for t in new_tokens]
    assert new_tokens[0].lemma_ == 'lemma'
    assert new_tokens[0].norm_ == 'norm'
    assert new_tokens[0].ent_kb_id_ == 'ent_kb_id'
    assert new_tokens[0].ent_id_ == 'ent_id'
    new_tokens = Doc(tokens.vocab).from_bytes(tokens.to_bytes(exclude=['tensor']), exclude=['tensor'])
    assert tokens.text == new_tokens.text
    assert [t.text for t in tokens] == [t.text for t in new_tokens]
    assert [t.orth for t in tokens] == [t.orth for t in new_tokens]
    new_tokens = Doc(tokens.vocab).from_bytes(tokens.to_bytes(exclude=['sentiment']), exclude=['sentiment'])
    assert tokens.text == new_tokens.text
    assert [t.text for t in tokens] == [t.text for t in new_tokens]
    assert [t.orth for t in tokens] == [t.orth for t in new_tokens]

    def inner_func(d1, d2):
        return 'hello!'
    _ = tokens.to_bytes()
    with pytest.warns(UserWarning):
        tokens.user_hooks['similarity'] = inner_func
        _ = tokens.to_bytes()