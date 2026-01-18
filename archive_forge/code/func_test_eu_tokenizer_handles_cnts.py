import pytest
@pytest.mark.parametrize('text,length', [('milesker ederra joan zen hitzaldia plazer hutsa', 7), ('astelehen guztia sofan pasau biot', 5)])
def test_eu_tokenizer_handles_cnts(eu_tokenizer, text, length):
    tokens = eu_tokenizer(text)
    assert len(tokens) == length