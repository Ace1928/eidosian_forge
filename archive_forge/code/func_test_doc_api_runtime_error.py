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
def test_doc_api_runtime_error(en_tokenizer):
    text = '67% of black households are single parent \n\n72% of all black babies born out of wedlock \n\n50% of all black kids don’t finish high school'
    deps = ['nummod', 'nsubj', 'prep', 'amod', 'pobj', 'ROOT', 'amod', 'attr', '', 'nummod', 'appos', 'prep', 'det', 'amod', 'pobj', 'acl', 'prep', 'prep', 'pobj', '', 'nummod', 'nsubj', 'prep', 'det', 'amod', 'pobj', 'aux', 'neg', 'ccomp', 'amod', 'dobj']
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], deps=deps)
    nps = []
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        if len(np) > 1:
            nps.append(np)
    with doc.retokenize() as retokenizer:
        for np in nps:
            attrs = {'tag': np.root.tag_, 'lemma': np.text, 'ent_type': np.root.ent_type_}
            retokenizer.merge(np, attrs=attrs)