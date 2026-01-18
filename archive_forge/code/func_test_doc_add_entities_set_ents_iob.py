import pytest
from spacy import registry
from spacy.pipeline import EntityRecognizer
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example
def test_doc_add_entities_set_ents_iob(en_vocab):
    text = ['This', 'is', 'a', 'lion']
    doc = Doc(en_vocab, words=text)
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    ner = EntityRecognizer(en_vocab, model)
    ner.initialize(lambda: [_ner_example(ner)])
    ner(doc)
    doc.ents = [('ANIMAL', 3, 4)]
    assert [w.ent_iob_ for w in doc] == ['O', 'O', 'O', 'B']
    doc.ents = [('WORD', 0, 2)]
    assert [w.ent_iob_ for w in doc] == ['B', 'I', 'O', 'O']