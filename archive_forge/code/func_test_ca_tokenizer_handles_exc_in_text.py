import pytest
def test_ca_tokenizer_handles_exc_in_text(ca_tokenizer):
    text = 'La Dra. Puig viu a la pl. dels Til·lers.'
    doc = ca_tokenizer(text)
    assert [t.text for t in doc] == ['La', 'Dra.', 'Puig', 'viu', 'a', 'la', 'pl.', 'd', 'els', 'Til·lers', '.']