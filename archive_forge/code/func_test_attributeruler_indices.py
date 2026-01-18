import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_indices(nlp):
    a = nlp.add_pipe('attribute_ruler')
    a.add([[{'ORTH': 'a'}, {'ORTH': 'test'}]], {'LEMMA': 'the', 'MORPH': 'Case=Nom|Number=Plur'}, index=0)
    a.add([[{'ORTH': 'This'}, {'ORTH': 'is'}]], {'LEMMA': 'was', 'MORPH': 'Case=Nom|Number=Sing'}, index=1)
    a.add([[{'ORTH': 'a'}, {'ORTH': 'test'}]], {'LEMMA': 'cat'}, index=-1)
    text = 'This is a test.'
    doc = nlp(text)
    for i in range(len(doc)):
        if i == 1:
            assert doc[i].lemma_ == 'was'
            assert str(doc[i].morph) == 'Case=Nom|Number=Sing'
        elif i == 2:
            assert doc[i].lemma_ == 'the'
            assert str(doc[i].morph) == 'Case=Nom|Number=Plur'
        elif i == 3:
            assert doc[i].lemma_ == 'cat'
        else:
            assert str(doc[i].morph) == ''
    a.add([[{'ORTH': 'a'}, {'ORTH': 'test'}]], {'LEMMA': 'cat'}, index=2)
    with pytest.raises(ValueError):
        doc = nlp(text)
    a.add([[{'ORTH': 'a'}, {'ORTH': 'test'}]], {'LEMMA': 'cat'}, index=10)
    with pytest.raises(ValueError):
        doc = nlp(text)