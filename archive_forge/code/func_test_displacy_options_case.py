import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_options_case():
    ents = ['foo', 'BAR']
    colors = {'FOO': 'red', 'bar': 'green'}
    renderer = EntityRenderer({'ents': ents, 'colors': colors})
    text = 'abcd'
    labels = ['foo', 'bar', 'FOO', 'BAR']
    spans = [{'start': i, 'end': i + 1, 'label': labels[i]} for i in range(len(text))]
    result = renderer.render_ents('abcde', spans, None).split('\n\n')
    assert 'red' in result[0] and 'foo' in result[0]
    assert 'green' in result[1] and 'bar' in result[1]
    assert 'red' in result[2] and 'FOO' in result[2]
    assert 'green' in result[3] and 'BAR' in result[3]