import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_parse_spans_different_spans_key(en_vocab):
    """Test that spans in a different spans key will be parsed"""
    doc = Doc(en_vocab, words=['Welcome', 'to', 'the', 'Bank', 'of', 'China'])
    doc.spans['sc'] = [Span(doc, 3, 6, 'ORG'), Span(doc, 5, 6, 'GPE')]
    doc.spans['custom'] = [Span(doc, 3, 6, 'BANK')]
    spans = displacy.parse_spans(doc, options={'spans_key': 'custom'})
    assert isinstance(spans, dict)
    assert spans['text'] == 'Welcome to the Bank of China '
    assert spans['spans'] == [{'start': 15, 'end': 28, 'start_token': 3, 'end_token': 6, 'label': 'BANK', 'kb_id': '', 'kb_url': '#'}]