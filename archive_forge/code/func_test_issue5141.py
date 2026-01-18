import pytest
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.tokens.underscore import Underscore
@pytest.mark.issue(5141)
def test_issue5141(en_vocab):
    """Ensure an empty DocBin does not crash on serialization"""
    doc_bin = DocBin(attrs=['DEP', 'HEAD'])
    assert list(doc_bin.get_docs(en_vocab)) == []
    doc_bin_bytes = doc_bin.to_bytes()
    doc_bin_2 = DocBin().from_bytes(doc_bin_bytes)
    assert list(doc_bin_2.get_docs(en_vocab)) == []