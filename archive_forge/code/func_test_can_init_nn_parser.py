import pytest
from thinc.api import Model
from spacy import registry
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.pipeline.transition_parser import Parser
from spacy.tokens.doc import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_can_init_nn_parser(parser):
    assert isinstance(parser.model, Model)