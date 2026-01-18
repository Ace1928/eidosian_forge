import pytest
import spacy
from spacy.training import loggers
@pytest.fixture()
def nlp():
    nlp = spacy.blank('en')
    nlp.add_pipe('ner')
    return nlp