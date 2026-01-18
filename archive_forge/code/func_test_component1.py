import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
@Language.component('c1', assigns=['token.tag', 'doc.tensor'])
def test_component1(doc):
    return doc