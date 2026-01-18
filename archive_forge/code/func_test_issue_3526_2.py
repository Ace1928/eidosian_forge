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
def test_issue_3526_2(en_vocab):
    patterns = [{'label': 'HELLO', 'pattern': 'hello world'}, {'label': 'BYE', 'pattern': [{'LOWER': 'bye'}, {'LOWER': 'bye'}]}, {'label': 'HELLO', 'pattern': [{'ORTH': 'HELLO'}]}, {'label': 'COMPLEX', 'pattern': [{'ORTH': 'foo', 'OP': '*'}]}, {'label': 'TECH_ORG', 'pattern': 'Apple', 'id': 'a1'}]
    nlp = Language(vocab=en_vocab)
    ruler = EntityRuler(nlp, patterns=patterns, overwrite_ents=True)
    bytes_old_style = srsly.msgpack_dumps(ruler.patterns)
    new_ruler = EntityRuler(nlp)
    new_ruler = new_ruler.from_bytes(bytes_old_style)
    assert len(new_ruler) == len(ruler)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert new_ruler.overwrite is not ruler.overwrite