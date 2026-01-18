import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
def load_tokenizer(b):
    tok = get_lang_class('en')().tokenizer
    tok.from_bytes(b)
    return tok