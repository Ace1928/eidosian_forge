import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
@registry.misc('attribute_ruler_patterns')
def attribute_ruler_patterns():
    return [{'patterns': [[{'ORTH': 'a'}], [{'ORTH': 'irrelevant'}]], 'attrs': {'LEMMA': 'the', 'MORPH': 'Case=Nom|Number=Plur'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'LEMMA': 'cat'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'MORPH': 'Case=Nom|Number=Sing'}, 'index': 0}]