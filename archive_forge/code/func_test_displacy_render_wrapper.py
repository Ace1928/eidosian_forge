import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_render_wrapper(en_vocab):
    """Test that displaCy accepts custom rendering wrapper."""

    def wrapper(html):
        return 'TEST' + html + 'TEST'
    displacy.set_render_wrapper(wrapper)
    doc = Doc(en_vocab, words=['But', 'Google', 'is', 'starting', 'from', 'behind'])
    doc.ents = [Span(doc, 1, 2, label=doc.vocab.strings['ORG'])]
    html = displacy.render(doc, style='ent')
    assert html.startswith('TEST<div')
    assert html.endswith('/div>TEST')
    displacy.set_render_wrapper(lambda html: html)