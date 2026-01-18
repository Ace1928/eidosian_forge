import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_render_manual_dep():
    """Test displacy.render with manual data for dep style"""
    parsed_dep = {'words': [{'text': 'This', 'tag': 'DT'}, {'text': 'is', 'tag': 'VBZ'}, {'text': 'a', 'tag': 'DT'}, {'text': 'sentence', 'tag': 'NN'}], 'arcs': [{'start': 0, 'end': 1, 'label': 'nsubj', 'dir': 'left'}, {'start': 2, 'end': 3, 'label': 'det', 'dir': 'left'}, {'start': 1, 'end': 3, 'label': 'attr', 'dir': 'right'}], 'title': 'Title'}
    html = displacy.render([parsed_dep], style='dep', manual=True)
    for word in parsed_dep['words']:
        assert word['text'] in html
        assert word['tag'] in html