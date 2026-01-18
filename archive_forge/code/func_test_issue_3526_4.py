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
def test_issue_3526_4(en_vocab):
    nlp = Language(vocab=en_vocab)
    patterns = [{'label': 'ORG', 'pattern': 'Apple'}]
    config = {'overwrite_ents': True}
    ruler = nlp.add_pipe('entity_ruler', config=config)
    ruler.add_patterns(patterns)
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        ruler = nlp.get_pipe('entity_ruler')
        assert ruler.patterns == [{'label': 'ORG', 'pattern': 'Apple'}]
        assert ruler.overwrite is True
        nlp2 = load(tmpdir)
        new_ruler = nlp2.get_pipe('entity_ruler')
        assert new_ruler.patterns == [{'label': 'ORG', 'pattern': 'Apple'}]
        assert new_ruler.overwrite is True