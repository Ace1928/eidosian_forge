from spacy.tokens import Doc
def test_en_parser_noun_chunks_appositional_modifiers(en_vocab):
    words = ['Sam', ',', 'my', 'brother', ',', 'arrived', 'to', 'the', 'house', '.']
    heads = [5, 0, 3, 0, 0, 5, 5, 8, 6, 5]
    pos = ['PROPN', 'PUNCT', 'DET', 'NOUN', 'PUNCT', 'VERB', 'ADP', 'DET', 'NOUN', 'PUNCT']
    deps = ['nsubj', 'punct', 'poss', 'appos', 'punct', 'ROOT', 'prep', 'det', 'pobj', 'punct']
    doc = Doc(en_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == 'Sam '
    assert chunks[1].text_with_ws == 'my brother '
    assert chunks[2].text_with_ws == 'the house '