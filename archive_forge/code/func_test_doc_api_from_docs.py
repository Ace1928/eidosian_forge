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
@pytest.mark.usefixtures('clean_underscore')
def test_doc_api_from_docs(en_tokenizer, de_tokenizer):
    en_texts = ['Merging the docs is fun.', '', "They don't think alike. ", '', 'Another doc.']
    en_texts_without_empty = [t for t in en_texts if len(t)]
    de_text = 'Wie war die Frage?'
    en_docs = [en_tokenizer(text) for text in en_texts]
    en_docs[0].spans['group'] = [en_docs[0][1:4]]
    en_docs[2].spans['group'] = [en_docs[2][1:4]]
    en_docs[4].spans['group'] = [en_docs[4][0:1]]
    span_group_texts = sorted([en_docs[0][1:4].text, en_docs[2][1:4].text, en_docs[4][0:1].text])
    de_doc = de_tokenizer(de_text)
    Token.set_extension('is_ambiguous', default=False)
    en_docs[0][2]._.is_ambiguous = True
    en_docs[2][3]._.is_ambiguous = True
    assert Doc.from_docs([]) is None
    assert de_doc is not Doc.from_docs([de_doc])
    assert str(de_doc) == str(Doc.from_docs([de_doc]))
    with pytest.raises(ValueError):
        Doc.from_docs(en_docs + [de_doc])
    m_doc = Doc.from_docs(en_docs)
    assert len(en_texts_without_empty) == len(list(m_doc.sents))
    assert len(m_doc.text) > len(en_texts[0]) + len(en_texts[1])
    assert m_doc.text == ' '.join([t.strip() for t in en_texts_without_empty])
    p_token = m_doc[len(en_docs[0]) - 1]
    assert p_token.text == '.' and bool(p_token.whitespace_)
    en_docs_tokens = [t for doc in en_docs for t in doc]
    assert len(m_doc) == len(en_docs_tokens)
    think_idx = len(en_texts[0]) + 1 + en_texts[2].index('think')
    assert m_doc[2]._.is_ambiguous is True
    assert m_doc[9].idx == think_idx
    assert m_doc[9]._.is_ambiguous is True
    assert not any([t._.is_ambiguous for t in m_doc[3:8]])
    assert 'group' in m_doc.spans
    assert span_group_texts == sorted([s.text for s in m_doc.spans['group']])
    assert bool(m_doc[11].whitespace_)
    m_doc = Doc.from_docs(en_docs, ensure_whitespace=False)
    assert len(en_texts_without_empty) == len(list(m_doc.sents))
    assert len(m_doc.text) == sum((len(t) for t in en_texts))
    assert m_doc.text == ''.join(en_texts_without_empty)
    p_token = m_doc[len(en_docs[0]) - 1]
    assert p_token.text == '.' and (not bool(p_token.whitespace_))
    en_docs_tokens = [t for doc in en_docs for t in doc]
    assert len(m_doc) == len(en_docs_tokens)
    think_idx = len(en_texts[0]) + 0 + en_texts[2].index('think')
    assert m_doc[9].idx == think_idx
    assert 'group' in m_doc.spans
    assert span_group_texts == sorted([s.text for s in m_doc.spans['group']])
    assert bool(m_doc[11].whitespace_)
    m_doc = Doc.from_docs(en_docs, attrs=['lemma', 'length', 'pos'])
    assert len(m_doc.text) > len(en_texts[0]) + len(en_texts[1])
    assert m_doc.text == ' '.join([t.strip() for t in en_texts_without_empty])
    p_token = m_doc[len(en_docs[0]) - 1]
    assert p_token.text == '.' and bool(p_token.whitespace_)
    en_docs_tokens = [t for doc in en_docs for t in doc]
    assert len(m_doc) == len(en_docs_tokens)
    think_idx = len(en_texts[0]) + 1 + en_texts[2].index('think')
    assert m_doc[9].idx == think_idx
    assert 'group' in m_doc.spans
    assert span_group_texts == sorted([s.text for s in m_doc.spans['group']])
    m_doc = Doc.from_docs(en_docs, exclude=['spans'])
    assert 'group' not in m_doc.spans
    m_doc = Doc.from_docs(en_docs, exclude=['user_data'])
    assert m_doc.user_data == {}
    doc = Doc.from_docs([en_tokenizer('')] * 10)
    en_docs = [en_tokenizer(text) for text in en_texts]
    m_doc = Doc.from_docs(en_docs)
    assert 'group' not in m_doc.spans
    for doc in en_docs:
        doc.spans['group'] = []
    m_doc = Doc.from_docs(en_docs)
    assert 'group' in m_doc.spans
    assert len(m_doc.spans['group']) == 0
    ops = get_current_ops()
    for doc in en_docs:
        doc.tensor = ops.asarray([[len(t.text), 0.0] for t in doc])
    m_doc = Doc.from_docs(en_docs)
    assert_array_equal(ops.to_numpy(m_doc.tensor), ops.to_numpy(ops.xp.vstack([doc.tensor for doc in en_docs if len(doc)])))
    m_doc = Doc.from_docs(en_docs, exclude=['tensor'])
    assert m_doc.tensor.shape == (0,)