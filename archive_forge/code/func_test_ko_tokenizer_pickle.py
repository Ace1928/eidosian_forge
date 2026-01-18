import pickle
from spacy.lang.ko import Korean
from ...util import make_tempdir
def test_ko_tokenizer_pickle(ko_tokenizer):
    b = pickle.dumps(ko_tokenizer)
    ko_tokenizer_re = pickle.loads(b)
    assert ko_tokenizer.to_bytes() == ko_tokenizer_re.to_bytes()