import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
@Language.component('c3', requires=['token.lemma'], assigns=['token._.custom_lemma'])
def test_component3(doc):
    return doc