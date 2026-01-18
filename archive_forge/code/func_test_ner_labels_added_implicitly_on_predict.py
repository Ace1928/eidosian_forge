import pytest
from thinc.api import Adam, fix_random_seed
from spacy import registry
from spacy.attrs import NORM
from spacy.language import Language
from spacy.pipeline import DependencyParser, EntityRecognizer
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_ner_labels_added_implicitly_on_predict():
    nlp = Language()
    ner = nlp.add_pipe('ner')
    for label in ['A', 'B', 'C']:
        ner.add_label(label)
    nlp.initialize()
    doc = Doc(nlp.vocab, words=['hello', 'world'], ents=['B-D', 'O'])
    ner(doc)
    assert [t.ent_type_ for t in doc] == ['D', '']
    assert 'D' in ner.labels