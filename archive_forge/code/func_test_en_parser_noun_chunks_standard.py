from spacy.tokens import Doc
def test_en_parser_noun_chunks_standard(en_vocab):
    words = ['A', 'base', 'phrase', 'should', 'be', 'recognized', '.']
    heads = [2, 2, 5, 5, 5, 5, 5]
    pos = ['DET', 'ADJ', 'NOUN', 'AUX', 'VERB', 'VERB', 'PUNCT']
    deps = ['det', 'amod', 'nsubjpass', 'aux', 'auxpass', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'A base phrase '