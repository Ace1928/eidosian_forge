from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
@pytest.fixture
def other_doc(en_tokenizer):
    doc = en_tokenizer('0 1 2 3 4 5 6')
    matcher = Matcher(en_tokenizer.vocab, validate=True)
    matcher.add('4', [[{}, {}, {}, {}]])
    matcher.add('2', [[{}, {}]])
    matcher.add('1', [[{}]])
    matches = matcher(doc)
    spans = []
    for match in matches:
        spans.append(Span(doc, match[1], match[2], en_tokenizer.vocab.strings[match[0]]))
    Random(42).shuffle(spans)
    doc.spans['SPANS'] = SpanGroup(doc, name='SPANS', attrs={'key': 'value'}, spans=spans)
    return doc