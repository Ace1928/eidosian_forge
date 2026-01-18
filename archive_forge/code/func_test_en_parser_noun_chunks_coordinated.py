from spacy.tokens import Doc
def test_en_parser_noun_chunks_coordinated(en_vocab):
    words = ['A', 'base', 'phrase', 'and', 'a', 'good', 'phrase', 'are', 'often', 'the', 'same', '.']
    heads = [2, 2, 7, 2, 6, 6, 2, 7, 7, 10, 7, 7]
    pos = ['DET', 'NOUN', 'NOUN', 'CCONJ', 'DET', 'ADJ', 'NOUN', 'VERB', 'ADV', 'DET', 'ADJ', 'PUNCT']
    deps = ['det', 'compound', 'nsubj', 'cc', 'det', 'amod', 'conj', 'ROOT', 'advmod', 'det', 'attr', 'punct']
    doc = Doc(en_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'A base phrase '
    assert chunks[1].text_with_ws == 'a good phrase '