import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_init(nlp, pattern_dicts):
    a = nlp.add_pipe('attribute_ruler')
    for p in pattern_dicts:
        a.add(**p)
    doc = nlp('This is a test.')
    assert doc[2].lemma_ == 'the'
    assert str(doc[2].morph) == 'Case=Nom|Number=Plur'
    assert doc[3].lemma_ == 'cat'
    assert str(doc[3].morph) == 'Case=Nom|Number=Sing'
    assert doc.has_annotation('LEMMA')
    assert doc.has_annotation('MORPH')