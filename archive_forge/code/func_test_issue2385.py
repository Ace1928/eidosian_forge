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
@pytest.mark.issue(2385)
def test_issue2385():
    """Test that IOB tags are correctly converted to BILUO tags."""
    tags1 = ('B-BRAWLER', 'I-BRAWLER', 'I-BRAWLER')
    assert iob_to_biluo(tags1) == ['B-BRAWLER', 'I-BRAWLER', 'L-BRAWLER']
    tags2 = ('I-ORG', 'I-ORG', 'B-ORG')
    assert iob_to_biluo(tags2) == ['B-ORG', 'L-ORG', 'U-ORG']
    tags3 = ('B-PERSON', 'I-PERSON', 'B-PERSON')
    assert iob_to_biluo(tags3) == ['B-PERSON', 'L-PERSON', 'U-PERSON']
    tags4 = ('B-MULTI-PERSON', 'I-MULTI-PERSON', 'B-MULTI-PERSON')
    assert iob_to_biluo(tags4) == ['B-MULTI-PERSON', 'L-MULTI-PERSON', 'U-MULTI-PERSON']