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
@pytest.mark.issue(2656)
def test_issue2656(en_tokenizer):
    """Test that tokenizer correctly splits off punctuation after numbers with
    decimal points.
    """
    doc = en_tokenizer('I went for 40.3, and got home by 10.0.')
    assert len(doc) == 11
    assert doc[0].text == 'I'
    assert doc[1].text == 'went'
    assert doc[2].text == 'for'
    assert doc[3].text == '40.3'
    assert doc[4].text == ','
    assert doc[5].text == 'and'
    assert doc[6].text == 'got'
    assert doc[7].text == 'home'
    assert doc[8].text == 'by'
    assert doc[9].text == '10.0'
    assert doc[10].text == '.'