import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_from_usage_docs(en_vocab):
    text = 'Wow 😀 This is really cool! 😂 😂'
    doc = Doc(en_vocab, words=text.split(' '))
    pos_emoji = ['😀', '😃', '😂', '🤣', '😊', '😍']
    pos_patterns = [[{'ORTH': emoji}] for emoji in pos_emoji]

    def label_sentiment(matcher, doc, i, matches):
        match_id, start, end = matches[i]
        if doc.vocab.strings[match_id] == 'HAPPY':
            doc.sentiment += 0.1
        span = doc[start:end]
        with doc.retokenize() as retokenizer:
            retokenizer.merge(span)
        token = doc[start]
        token.vocab[token.text].norm_ = 'happy emoji'
    matcher = Matcher(en_vocab)
    matcher.add('HAPPY', pos_patterns, on_match=label_sentiment)
    matcher(doc)
    assert doc.sentiment != 0
    assert doc[1].norm_ == 'happy emoji'