import pytest
from spacy.morphology import Morphology
from spacy.strings import StringStore, get_string_id
def test_add_morphology_with_string_names(morphology):
    morphology.add({'Case': 'gen', 'Number': 'sing'})