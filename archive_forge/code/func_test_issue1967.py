import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('label', ['U-JOB-NAME'])
@pytest.mark.issue(1967)
def test_issue1967(label):
    nlp = Language()
    config = {}
    ner = nlp.create_pipe('ner', config=config)
    example = Example.from_dict(Doc(ner.vocab, words=['word']), {'ids': [0], 'words': ['word'], 'tags': ['tag'], 'heads': [0], 'deps': ['dep'], 'entities': [label]})
    assert 'JOB-NAME' in ner.moves.get_actions(examples=[example])[1]