import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string  # type: ignore
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
def test_floret_vectors(floret_vectors_vec_str, floret_vectors_hashvec_str):
    nlp = English()
    nlp_plain = English()
    with make_tempdir() as tmpdir:
        p = tmpdir / 'test.hashvec'
        with open(p, 'w') as fileh:
            fileh.write(floret_vectors_hashvec_str)
        convert_vectors(nlp, p, truncate=0, prune=-1, mode='floret')
        p = tmpdir / 'test.vec'
        with open(p, 'w') as fileh:
            fileh.write(floret_vectors_vec_str)
        convert_vectors(nlp_plain, p, truncate=0, prune=-1)
    word = 'der'
    ngrams = nlp.vocab.vectors._get_ngrams(word)
    assert ngrams == ['<der>', '<d', 'de', 'er', 'r>', '<de', 'der', 'er>']
    rows = OPS.xp.asarray([h % nlp.vocab.vectors.shape[0] for ngram in ngrams for h in nlp.vocab.vectors._get_ngram_hashes(ngram)], dtype='uint32')
    assert_equal(OPS.to_numpy(rows), numpy.asarray([5, 6, 7, 5, 8, 2, 8, 9, 3, 3, 4, 6, 7, 3, 0, 2]))
    assert len(rows) == len(ngrams) * nlp.vocab.vectors.hash_count
    for word in nlp_plain.vocab.vectors:
        word = nlp_plain.vocab.strings.as_string(word)
        assert_almost_equal(nlp.vocab[word].vector, nlp_plain.vocab[word].vector, decimal=3)
        assert nlp.vocab[word * 5].has_vector
    assert nlp_plain.vocab.vectors.n_keys > 0
    assert nlp.vocab.vectors.n_keys == -1
    words = [s for s in nlp_plain.vocab.vectors]
    single_vecs = OPS.to_numpy(OPS.asarray([nlp.vocab[word].vector for word in words]))
    batch_vecs = OPS.to_numpy(nlp.vocab.vectors.get_batch(words))
    assert_equal(single_vecs, batch_vecs)
    assert_equal(OPS.to_numpy(nlp.vocab[''].vector), numpy.zeros((nlp.vocab.vectors.shape[0],)))
    assert_equal(OPS.to_numpy(nlp.vocab.vectors.get_batch([''])), numpy.zeros((1, nlp.vocab.vectors.shape[0])))
    assert_equal(OPS.to_numpy(nlp.vocab.vectors.get_batch(['a', '', 'b'])[1]), numpy.zeros((nlp.vocab.vectors.shape[0],)))
    vector = list(range(nlp.vocab.vectors.shape[1]))
    orig_bytes = nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.set_vector('the', vector)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab[word].vector = vector
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.add('the', row=6)
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.warns(UserWarning):
        nlp.vocab.vectors.resize(shape=(100, 10))
    assert orig_bytes == nlp.vocab.vectors.to_bytes(exclude=['strings'])
    with pytest.raises(ValueError):
        nlp.vocab.vectors.clear()
    with make_tempdir() as d:
        nlp.vocab.to_disk(d)
        vocab_r = Vocab()
        vocab_r.from_disk(d)
        assert nlp.vocab.vectors.to_bytes() == vocab_r.vectors.to_bytes()
        assert_equal(OPS.to_numpy(nlp.vocab.vectors.data), OPS.to_numpy(vocab_r.vectors.data))
        assert_equal(nlp.vocab.vectors._get_cfg(), vocab_r.vectors._get_cfg())
        assert_almost_equal(OPS.to_numpy(nlp.vocab[word].vector), OPS.to_numpy(vocab_r[word].vector), decimal=6)