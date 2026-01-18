import pytest
@pytest.mark.parametrize('text,lemma', [('aprox.', 'aproximadament'), ('pàg.', 'pàgina'), ('p.ex.', 'per exemple')])
def test_ca_tokenizer_handles_abbr(ca_tokenizer, text, lemma):
    tokens = ca_tokenizer(text)
    assert len(tokens) == 1