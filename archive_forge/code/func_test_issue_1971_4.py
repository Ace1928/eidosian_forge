import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(1971)
def test_issue_1971_4(en_vocab):
    """Test that pattern matches correctly with multiple extension attribute
    values on a single token.
    """
    Token.set_extension('ext_a', default='str_a', force=True)
    Token.set_extension('ext_b', default='str_b', force=True)
    matcher = Matcher(en_vocab)
    doc = Doc(en_vocab, words=['this', 'is', 'text'])
    pattern = [{'_': {'ext_a': 'str_a', 'ext_b': 'str_b'}}] * 3
    matcher.add('TEST', [pattern])
    matches = matcher(doc)
    assert len(matches) == 1
    assert matches[0] == (en_vocab.strings['TEST'], 0, 3)