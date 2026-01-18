import pytest
def test_ne_tokenizer_handlers_long_text(ne_tokenizer):
    text = 'मैले पाएको सर्टिफिकेटलाई म त बोक्रो सम्झन्छु र अभ्यास तब सुरु भयो, जब मैले कलेज पार गरेँ र जीवनको पढाइ सुरु गरेँ ।'
    tokens = ne_tokenizer(text)
    assert len(tokens) == 24