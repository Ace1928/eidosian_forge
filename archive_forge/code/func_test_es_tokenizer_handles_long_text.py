import pytest
from spacy.lang.es import Spanish
from spacy.lang.es.lex_attrs import like_num
def test_es_tokenizer_handles_long_text(es_tokenizer):
    text = 'Cuando a José Mujica lo invitaron a dar una conferencia\n\nen Oxford este verano, su cabeza hizo "crac". La "más antigua" universidad de habla\n\ninglesa, esa que cobra decenas de miles de euros de matrícula a sus alumnos\n\ny en cuyos salones han disertado desde Margaret Thatcher hasta Stephen Hawking,\n\nreclamaba los servicios de este viejo de 81 años, formado en un colegio público\n\nen Montevideo y que pregona las bondades de la vida austera.'
    tokens = es_tokenizer(text)
    assert len(tokens) == 90