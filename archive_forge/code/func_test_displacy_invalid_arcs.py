import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_invalid_arcs():
    renderer = DependencyRenderer()
    words = [{'text': 'This', 'tag': 'DET'}, {'text': 'is', 'tag': 'VERB'}]
    arcs = [{'start': 0, 'end': 1, 'label': 'nsubj', 'dir': 'left'}, {'start': -1, 'end': 2, 'label': 'det', 'dir': 'left'}]
    with pytest.raises(ValueError):
        renderer.render([{'words': words, 'arcs': arcs}])