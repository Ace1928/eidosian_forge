import pytest
@pytest.mark.xfail
def test_indefinite_article(af_tokenizer):
    text = "as 'n algemene standaard"
    tokens = af_tokenizer(text)
    assert len(tokens) == 4