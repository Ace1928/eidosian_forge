from spacy.tokens import Doc
def test_de_extended_chunk(de_vocab):
    words = ['Die', 'Sängerin', 'singt', 'mit', 'einer', 'Tasse', 'Kaffee', 'Arien', '.']
    heads = [1, 2, 2, 2, 5, 3, 5, 2, 2]
    pos = ['DET', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN', 'NOUN', 'NOUN', 'PUNCT']
    deps = ['nk', 'sb', 'ROOT', 'mo', 'nk', 'nk', 'nk', 'oa', 'punct']
    doc = Doc(de_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == 'Die Sängerin '
    assert chunks[1].text_with_ws == 'einer Tasse Kaffee '
    assert chunks[2].text_with_ws == 'Arien '