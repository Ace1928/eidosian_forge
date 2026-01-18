import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def label_sentiment(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    if doc.vocab.strings[match_id] == 'HAPPY':
        doc.sentiment += 0.1
    span = doc[start:end]
    with doc.retokenize() as retokenizer:
        retokenizer.merge(span)
    token = doc[start]
    token.vocab[token.text].norm_ = 'happy emoji'