import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_init_patterns(nlp, pattern_dicts):
    ruler = nlp.add_pipe('attribute_ruler')
    ruler.initialize(lambda: [], patterns=pattern_dicts)
    doc = nlp('This is a test.')
    assert doc[2].lemma_ == 'the'
    assert str(doc[2].morph) == 'Case=Nom|Number=Plur'
    assert doc[3].lemma_ == 'cat'
    assert str(doc[3].morph) == 'Case=Nom|Number=Sing'
    assert doc.has_annotation('LEMMA')
    assert doc.has_annotation('MORPH')
    nlp.remove_pipe('attribute_ruler')

    @registry.misc('attribute_ruler_patterns')
    def attribute_ruler_patterns():
        return [{'patterns': [[{'ORTH': 'a'}], [{'ORTH': 'irrelevant'}]], 'attrs': {'LEMMA': 'the', 'MORPH': 'Case=Nom|Number=Plur'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'LEMMA': 'cat'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'MORPH': 'Case=Nom|Number=Sing'}, 'index': 0}]
    nlp.config['initialize']['components']['attribute_ruler'] = {'patterns': {'@misc': 'attribute_ruler_patterns'}}
    nlp.add_pipe('attribute_ruler')
    nlp.initialize()
    doc = nlp('This is a test.')
    assert doc[2].lemma_ == 'the'
    assert str(doc[2].morph) == 'Case=Nom|Number=Plur'
    assert doc[3].lemma_ == 'cat'
    assert str(doc[3].morph) == 'Case=Nom|Number=Sing'
    assert doc.has_annotation('LEMMA')
    assert doc.has_annotation('MORPH')