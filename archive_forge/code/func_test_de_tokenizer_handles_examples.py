import pytest
@pytest.mark.parametrize('text,length', [('»Was ist mit mir geschehen?«, dachte er.', 12), ('“Dies frühzeitige Aufstehen”, dachte er, “macht einen ganz blödsinnig. ', 15)])
def test_de_tokenizer_handles_examples(de_tokenizer, text, length):
    tokens = de_tokenizer(text)
    assert len(tokens) == length