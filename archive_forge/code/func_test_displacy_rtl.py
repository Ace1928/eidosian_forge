import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_rtl():
    words = ['ما', 'بسیار', 'کتاب', 'می\u200cخوانیم']
    pos = ['PRO', 'ADV', 'N_PL', 'V_SUB']
    deps = ['foo', 'bar', 'foo', 'baz']
    heads = [1, 0, 3, 1]
    nlp = Persian()
    doc = Doc(nlp.vocab, words=words, tags=pos, heads=heads, deps=deps)
    doc.ents = [Span(doc, 1, 3, label='TEST')]
    html = displacy.render(doc, page=True, style='dep')
    assert 'direction: rtl' in html
    assert 'direction="rtl"' in html
    assert f'lang="{nlp.lang}"' in html
    html = displacy.render(doc, page=True, style='ent')
    assert 'direction: rtl' in html
    assert f'lang="{nlp.lang}"' in html