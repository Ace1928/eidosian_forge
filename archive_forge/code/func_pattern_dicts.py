import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
@pytest.fixture
def pattern_dicts():
    return [{'patterns': [[{'ORTH': 'a'}], [{'ORTH': 'irrelevant'}]], 'attrs': {'LEMMA': 'the', 'MORPH': 'Case=Nom|Number=Plur'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'LEMMA': 'cat'}}, {'patterns': [[{'ORTH': 'test'}]], 'attrs': {'MORPH': 'Case=Nom|Number=Sing'}, 'index': 0}]