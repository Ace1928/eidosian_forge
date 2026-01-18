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
def test_doc_set_ents(en_tokenizer):
    doc = en_tokenizer('a b c d e')
    doc.set_ents([Span(doc, 0, 1, 10), Span(doc, 1, 3, 11)])
    assert [t.ent_iob for t in doc] == [3, 3, 1, 2, 2]
    assert [t.ent_type for t in doc] == [10, 11, 11, 0, 0]
    doc = en_tokenizer('a b c d e')
    doc.set_ents([Span(doc, 0, 1, 10), Span(doc, 1, 3, 11)])
    doc.set_ents([Span(doc, 0, 2, 12)], default='unmodified')
    assert [t.ent_iob for t in doc] == [3, 1, 3, 2, 2]
    assert [t.ent_type for t in doc] == [12, 12, 11, 0, 0]
    doc = en_tokenizer('a b c d e')
    doc.set_ents([Span(doc, 0, 1, 10), Span(doc, 1, 3, 11)], missing=[doc[4:5]])
    assert [t.ent_iob for t in doc] == [3, 3, 1, 2, 0]
    assert [t.ent_type for t in doc] == [10, 11, 11, 0, 0]
    doc = en_tokenizer('a b c d e')
    doc.set_ents([Span(doc, 0, 1, 10), Span(doc, 1, 3, 11)], outside=[doc[4:5]], default='missing')
    assert [t.ent_iob for t in doc] == [3, 3, 1, 0, 2]
    assert [t.ent_type for t in doc] == [10, 11, 11, 0, 0]
    doc = en_tokenizer('a b c d e')
    doc.set_ents([], blocked=[doc[1:2], doc[3:5]], default='unmodified')
    assert [t.ent_iob for t in doc] == [0, 3, 0, 3, 3]
    assert [t.ent_type for t in doc] == [0, 0, 0, 0, 0]
    assert doc.ents == tuple()
    doc.ents = [Span(doc, 3, 5, 'ENT')]
    assert [t.ent_iob for t in doc] == [2, 2, 2, 3, 1]
    doc.set_ents([], blocked=[doc[3:4]], default='unmodified')
    assert [t.ent_iob for t in doc] == [2, 2, 2, 3, 3]
    doc = en_tokenizer('a b c d e')
    doc.set_ents([Span(doc, 0, 1, 10)], blocked=[doc[1:2]], missing=[doc[2:3]], outside=[doc[3:4]], default='unmodified')
    assert [t.ent_iob for t in doc] == [3, 3, 0, 2, 0]
    assert [t.ent_type for t in doc] == [10, 0, 0, 0, 0]
    doc = en_tokenizer('a b c d e')
    with pytest.raises(ValueError):
        doc.set_ents([], missing=doc[1:2])
    with pytest.raises(ValueError):
        doc.set_ents([], missing=[doc[1:2]], default='none')
    with pytest.raises(ValueError):
        doc.set_ents([], missing=[doc[1:2]], outside=[doc[1:2]])