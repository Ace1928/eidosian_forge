import pickle
from spacy.lang.ja import Japanese
from ...util import make_tempdir
def test_ja_tokenizer_pickle(ja_tokenizer):
    b = pickle.dumps(ja_tokenizer)
    ja_tokenizer_re = pickle.loads(b)
    assert ja_tokenizer.to_bytes() == ja_tokenizer_re.to_bytes()