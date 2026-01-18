import pickle
import re
import pytest
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.training import Example
from spacy.util import load_config_from_str
from ..util import make_tempdir
@pytest.mark.issue(6950)
def test_issue6950():
    """Test that the nlp object with initialized tok2vec with listeners pickles
    correctly (and doesn't have lambdas).
    """
    nlp = English.from_config(load_config_from_str(CONFIG_ISSUE_6950))
    nlp.initialize(lambda: [Example.from_dict(nlp.make_doc('hello'), {'tags': ['V']})])
    pickle.dumps(nlp)
    nlp('hello')
    pickle.dumps(nlp)