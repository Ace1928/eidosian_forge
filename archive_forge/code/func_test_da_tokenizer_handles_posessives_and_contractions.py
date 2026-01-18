import pytest
def test_da_tokenizer_handles_posessives_and_contractions(da_tokenizer):
    tokens = da_tokenizer("'DBA's, Lars' og Liz' bil sku' sgu' ik' ha' en bule, det ka' han ik' li' mere', sagde hun.")
    assert len(tokens) == 25
    assert tokens[0].text == "'"
    assert tokens[1].text == "DBA's"
    assert tokens[2].text == ','
    assert tokens[3].text == "Lars'"
    assert tokens[4].text == 'og'
    assert tokens[5].text == "Liz'"
    assert tokens[6].text == 'bil'
    assert tokens[7].text == "sku'"
    assert tokens[8].text == "sgu'"
    assert tokens[9].text == "ik'"
    assert tokens[10].text == "ha'"
    assert tokens[11].text == 'en'
    assert tokens[12].text == 'bule'
    assert tokens[13].text == ','
    assert tokens[14].text == 'det'
    assert tokens[15].text == "ka'"
    assert tokens[16].text == 'han'
    assert tokens[17].text == "ik'"
    assert tokens[18].text == "li'"
    assert tokens[19].text == 'mere'
    assert tokens[20].text == "'"
    assert tokens[21].text == ','
    assert tokens[22].text == 'sagde'
    assert tokens[23].text == 'hun'
    assert tokens[24].text == '.'