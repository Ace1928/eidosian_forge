import pytest
def test_ko_empty_doc(ko_tokenizer):
    tokens = ko_tokenizer('')
    assert len(tokens) == 0