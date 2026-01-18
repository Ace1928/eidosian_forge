import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(12816)
def test_issue12816(en_vocab) -> None:
    """Test that displaCy's span visualizer escapes annotated HTML tags correctly."""
    doc = Doc(en_vocab, words=['test', '<TEST>'])
    doc.spans['sc'] = [Span(doc, 0, 1, label='test')]
    html = displacy.render(doc, style='span')
    assert '&lt;TEST&gt;' in html
    doc.spans['sc'].append(Span(doc, 1, 2, label='test'))
    html = displacy.render(doc, style='span')
    assert '&lt;TEST&gt;' in html