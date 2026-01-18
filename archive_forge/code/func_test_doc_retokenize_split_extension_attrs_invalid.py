import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
@pytest.mark.parametrize('underscore_attrs', [[{'a': 'x'}, {}], [{'b': 'x'}, {}], [{'c': 'x'}, {}], [{'a': 'x'}, {'x': 'x'}], [{'a': 'x', 'x': 'x'}, {'x': 'x'}], {'x': 'x'}])
def test_doc_retokenize_split_extension_attrs_invalid(en_vocab, underscore_attrs):
    Token.set_extension('x', default=False, force=True)
    Token.set_extension('a', getter=lambda x: x, force=True)
    Token.set_extension('b', method=lambda x: x, force=True)
    doc = Doc(en_vocab, words=['LosAngeles', 'start'])
    attrs = {'_': underscore_attrs}
    with pytest.raises(ValueError):
        with doc.retokenize() as retokenizer:
            heads = [(doc[0], 1), doc[1]]
            retokenizer.split(doc[0], ['Los', 'Angeles'], heads, attrs=attrs)