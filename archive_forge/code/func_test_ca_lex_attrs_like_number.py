import pytest
@pytest.mark.parametrize('text,match', [('10', True), ('1', True), ('10,000', True), ('10,00', True), ('999.0', True), ('un', True), ('dos', True), ('bilió', True), ('gos', False), (',', False), ('1/2', True)])
def test_ca_lex_attrs_like_number(ca_tokenizer, text, match):
    tokens = ca_tokenizer(text)
    assert len(tokens) == 1
    assert tokens[0].like_num == match