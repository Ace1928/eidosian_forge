import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_render_manual_span():
    """Test displacy.render with manual data for span style"""
    parsed_spans = [{'text': 'Welcome to the Bank of China.', 'spans': [{'start_token': 3, 'end_token': 6, 'label': 'ORG'}, {'start_token': 5, 'end_token': 6, 'label': 'GPE'}], 'tokens': ['Welcome', 'to', 'the', 'Bank', 'of', 'China', '.']}, {'text': 'Welcome to the Bank of China.', 'spans': [{'start_token': 3, 'end_token': 6, 'label': 'ORG'}, {'start_token': 5, 'end_token': 6, 'label': 'GPE'}], 'tokens': ['Welcome', 'to', 'the', 'Bank', 'of', 'China', '.'], 'title': 'Title'}]
    html = displacy.render(parsed_spans, style='span', manual=True)
    for parsed_span in parsed_spans:
        assert parsed_span['spans'][0]['label'] in html
        if 'title' in parsed_span:
            assert parsed_span['title'] in html