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
def test_build_model(parser, vocab):
    config = {'learn_tokens': False, 'min_action_freq': 0, 'update_with_oracle_cut_size': 100}
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser.model = Parser(vocab, model=model, moves=parser.moves, **config).model
    assert parser.model is not None