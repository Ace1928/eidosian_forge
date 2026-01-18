import pytest
from spacy.tokens import Doc
def test_en_noun_chunks_not_nested(doc, en_vocab):
    """Test that each token only appears in one noun chunk at most"""
    word_occurred = {}
    chunks = list(doc.noun_chunks)
    assert len(chunks) > 1
    for chunk in chunks:
        for word in chunk:
            word_occurred.setdefault(word.text, 0)
            word_occurred[word.text] += 1
    assert len(word_occurred) > 0
    for word, freq in word_occurred.items():
        assert freq == 1, (word, [chunk.text for chunk in doc.noun_chunks])