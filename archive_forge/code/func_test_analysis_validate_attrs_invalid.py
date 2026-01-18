import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
@pytest.mark.parametrize('attr', ['doc', 'doc_ents', 'doc.xyz', 'token.xyz', 'token.tag_', 'token.tag.xyz', 'token._.xyz.abc', 'span.label'])
def test_analysis_validate_attrs_invalid(attr):
    with pytest.raises(ValueError):
        validate_attrs([attr])