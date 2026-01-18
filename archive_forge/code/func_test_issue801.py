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
@pytest.mark.issue(801)
@pytest.mark.skip(reason='Can not be fixed unless with variable-width lookbehinds, cf. PR #3218')
@pytest.mark.parametrize('text,tokens', [('"deserve,"--and', ['"', 'deserve', ',"--', 'and']), ('exception;--exclusive', ['exception', ';--', 'exclusive']), ('day.--Is', ['day', '.--', 'Is']), ('refinement:--just', ['refinement', ':--', 'just']), ('memories?--To', ['memories', '?--', 'To']), ('Useful.=--Therefore', ['Useful', '.=--', 'Therefore']), ('=Hope.=--Pandora', ['=', 'Hope', '.=--', 'Pandora'])])
def test_issue801(en_tokenizer, text, tokens):
    """Test that special characters + hyphens are split correctly."""
    doc = en_tokenizer(text)
    assert len(doc) == len(tokens)
    assert [t.text for t in doc] == tokens