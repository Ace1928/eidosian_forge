import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(2361)
def test_issue2361(de_vocab):
    """Test if < is escaped when rendering"""
    chars = ('&lt;', '&gt;', '&amp;', '&quot;')
    words = ['<', '>', '&', '"']
    doc = Doc(de_vocab, words=words, deps=['dep'] * len(words))
    html = displacy.render(doc)
    for char in chars:
        assert char in html