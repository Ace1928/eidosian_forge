import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
@pytest.mark.parametrize('obj', [Doc, Span, Token])
def test_doc_underscore_remove_extension(obj):
    ext_name = 'to_be_removed'
    obj.set_extension(ext_name, default=False)
    assert obj.has_extension(ext_name)
    obj.remove_extension(ext_name)
    assert not obj.has_extension(ext_name)