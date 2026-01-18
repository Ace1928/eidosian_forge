import pytest
@pytest.mark.parametrize('text,length', [('समय जान कति पनि बेर लाग्दैन ।', 7), ('म ठूलो हुँदै थिएँ ।', 5)])
def test_ne_tokenizer_handles_cnts(ne_tokenizer, text, length):
    tokens = ne_tokenizer(text)
    assert len(tokens) == length