import pytest
from spacy.morphology import Morphology
from spacy.strings import StringStore, get_string_id
def test_morphology_tags_hash_distinctly(morphology):
    tag1 = morphology.add({'PunctSide': 'ini', 'VerbType': 'aux'})
    tag2 = morphology.add({'Case': 'gen', 'Number': 'sing'})
    assert tag1 != tag2