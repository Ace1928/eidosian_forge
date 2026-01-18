import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
@pytest.mark.skip(reason='Can not be fixed without iterative looping between prefix/suffix and infix')
@pytest.mark.issue(2070)
def test_issue2070():
    """Test that checks that a dot followed by a quote is handled
    appropriately.
    """
    nlp = English()
    doc = nlp('First sentence."A quoted sentence" he said ...')
    assert len(doc) == 11