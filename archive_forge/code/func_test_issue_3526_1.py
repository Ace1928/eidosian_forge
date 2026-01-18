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
@pytest.mark.issue(3526)
def test_issue_3526_1(en_vocab):
    patterns = [{'label': 'HELLO', 'pattern': 'hello world'}, {'label': 'BYE', 'pattern': [{'LOWER': 'bye'}, {'LOWER': 'bye'}]}, {'label': 'HELLO', 'pattern': [{'ORTH': 'HELLO'}]}, {'label': 'COMPLEX', 'pattern': [{'ORTH': 'foo', 'OP': '*'}]}, {'label': 'TECH_ORG', 'pattern': 'Apple', 'id': 'a1'}]
    nlp = Language(vocab=en_vocab)
    ruler = EntityRuler(nlp, patterns=patterns, overwrite_ents=True)
    ruler_bytes = ruler.to_bytes()
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    assert ruler.overwrite
    new_ruler = EntityRuler(nlp)
    new_ruler = new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler) == len(ruler)
    assert len(new_ruler.labels) == 4
    assert new_ruler.overwrite == ruler.overwrite
    assert new_ruler.ent_id_sep == ruler.ent_id_sep