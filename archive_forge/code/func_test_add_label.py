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
def test_add_label(parser):
    parser = _train_parser(parser)
    parser.add_label('right')
    sgd = Adam(0.001)
    for i in range(100):
        losses = {}
        parser.update([_parser_example(parser)], sgd=sgd, losses=losses)
    doc = Doc(parser.vocab, words=['a', 'b', 'c', 'd'])
    doc = parser(doc)
    assert doc[0].dep_ == 'right'
    assert doc[2].dep_ == 'left'