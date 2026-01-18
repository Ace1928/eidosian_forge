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
@pytest.mark.issue(4042)
def test_issue4042():
    """Test that serialization of an EntityRuler before NER works fine."""
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('SOME_LABEL')
    nlp.initialize()
    patterns = [{'label': 'MY_ORG', 'pattern': 'Apple'}, {'label': 'MY_GPE', 'pattern': [{'lower': 'san'}, {'lower': 'francisco'}]}]
    ruler = nlp.add_pipe('entity_ruler', before='ner')
    ruler.add_patterns(patterns)
    doc1 = nlp('What do you think about Apple ?')
    assert doc1.ents[0].label_ == 'MY_ORG'
    with make_tempdir() as d:
        output_dir = ensure_path(d)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        nlp2 = load_model(output_dir)
        doc2 = nlp2('What do you think about Apple ?')
        assert doc2.ents[0].label_ == 'MY_ORG'