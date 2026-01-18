import pytest
from spacy import registry
from spacy.pipeline import EntityRecognizer
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example
def test_add_overlapping_entities(en_vocab):
    text = ['Louisiana', 'Office', 'of', 'Conservation']
    doc = Doc(en_vocab, words=text)
    entity = Span(doc, 0, 4, label=391)
    doc.ents = [entity]
    new_entity = Span(doc, 0, 1, label=392)
    with pytest.raises(ValueError):
        doc.ents = list(doc.ents) + [new_entity]