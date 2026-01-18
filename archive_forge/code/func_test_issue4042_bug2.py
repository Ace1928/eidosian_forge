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
def test_issue4042_bug2():
    """
    Test that serialization of an NER works fine when new labels were added.
    This is the second bug of two bugs underlying the issue 4042.
    """
    nlp1 = English()
    ner1 = nlp1.add_pipe('ner')
    ner1.add_label('SOME_LABEL')
    nlp1.initialize()
    doc1 = nlp1('What do you think about Apple ?')
    assert len(ner1.labels) == 1
    assert 'SOME_LABEL' in ner1.labels
    apple_ent = Span(doc1, 5, 6, label='MY_ORG')
    doc1.ents = list(doc1.ents) + [apple_ent]
    ner1.add_label('MY_ORG')
    ner1(doc1)
    assert len(ner1.labels) == 2
    assert 'SOME_LABEL' in ner1.labels
    assert 'MY_ORG' in ner1.labels
    with make_tempdir() as d:
        output_dir = ensure_path(d)
        if not output_dir.exists():
            output_dir.mkdir()
        ner1.to_disk(output_dir)
        config = {}
        ner2 = nlp1.create_pipe('ner', config=config)
        ner2.from_disk(output_dir)
        assert len(ner2.labels) == 2