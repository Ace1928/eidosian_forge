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
@pytest.mark.issue(2926)
def test_issue2926(fr_tokenizer):
    """Test that the tokenizer correctly splits tokens separated by a slash (/)
    ending in a digit.
    """
    doc = fr_tokenizer('Learn html5/css3/javascript/jquery')
    assert len(doc) == 8
    assert doc[0].text == 'Learn'
    assert doc[1].text == 'html5'
    assert doc[2].text == '/'
    assert doc[3].text == 'css3'
    assert doc[4].text == '/'
    assert doc[5].text == 'javascript'
    assert doc[6].text == '/'
    assert doc[7].text == 'jquery'