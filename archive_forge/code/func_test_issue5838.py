import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(5838)
def test_issue5838():
    sample_text = 'First line\nSecond line, with ent\nThird line\nFourth line\n'
    nlp = English()
    doc = nlp(sample_text)
    doc.ents = [Span(doc, 7, 8, label='test')]
    html = displacy.render(doc, style='ent')
    found = html.count('<br>')
    assert found == 4