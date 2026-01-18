import pytest
from spacy import registry
from spacy.pipeline import EntityRecognizer
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example
def test_ents_reset(en_vocab):
    """Ensure that resetting doc.ents does not change anything"""
    text = ['This', 'is', 'a', 'lion']
    doc = Doc(en_vocab, words=text)
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    ner = EntityRecognizer(en_vocab, model)
    ner.initialize(lambda: [_ner_example(ner)])
    ner(doc)
    orig_iobs = [t.ent_iob_ for t in doc]
    doc.ents = list(doc.ents)
    assert [t.ent_iob_ for t in doc] == orig_iobs