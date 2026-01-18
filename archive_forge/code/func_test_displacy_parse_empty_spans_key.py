import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_parse_empty_spans_key(en_vocab):
    """Test that having an unset spans key doesn't raise an error"""
    doc = Doc(en_vocab, words=['Welcome', 'to', 'the', 'Bank', 'of', 'China'])
    doc.spans['custom'] = [Span(doc, 3, 6, 'BANK')]
    with pytest.warns(UserWarning, match='W117'):
        spans = displacy.parse_spans(doc)
    assert isinstance(spans, dict)