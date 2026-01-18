import pickle
import pytest
import srsly
from thinc.api import Linear
import spacy
from spacy import Vocab, load, registry
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import (
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.tokens import Span
from spacy.util import ensure_path, load_model
from ..util import make_tempdir
@pytest.fixture
def blank_parser(en_vocab):
    config = {'learn_tokens': False, 'min_action_freq': 30, 'update_with_oracle_cut_size': 100, 'beam_width': 1, 'beam_update_prob': 1.0, 'beam_density': 0.0}
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser = DependencyParser(en_vocab, model, **config)
    return parser