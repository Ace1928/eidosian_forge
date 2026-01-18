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
def test_load_without_strings():
    nlp = spacy.blank('en')
    orig_strings_length = len(nlp.vocab.strings)
    word = 'unlikely_word_' * 20
    nlp.vocab.strings.add(word)
    assert len(nlp.vocab.strings) == orig_strings_length + 1
    with make_tempdir() as d:
        nlp.to_disk(d)
        reloaded_nlp = load(d)
        assert len(nlp.vocab.strings) == len(reloaded_nlp.vocab.strings)
        assert word in reloaded_nlp.vocab.strings
        reloaded_nlp = load(d, exclude=['strings'])
        assert orig_strings_length == len(reloaded_nlp.vocab.strings)
        assert word not in reloaded_nlp.vocab.strings