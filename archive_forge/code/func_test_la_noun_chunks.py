import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('text,pos,deps,heads,expected_noun_chunks', LA_NP_TEST_EXAMPLES)
def test_la_noun_chunks(la_tokenizer, text, pos, deps, heads, expected_noun_chunks):
    tokens = la_tokenizer(text)
    assert len(heads) == len(pos)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=[head + i for i, head in enumerate(heads)], deps=deps, pos=pos)
    noun_chunks = list(doc.noun_chunks)
    assert len(noun_chunks) == len(expected_noun_chunks)
    for i, np in enumerate(noun_chunks):
        assert np.text == expected_noun_chunks[i]