import pickle
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.util import ensure_path, load_model
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(599)
def test_issue599(en_vocab):
    doc = Doc(en_vocab)
    doc2 = Doc(doc.vocab)
    doc2.from_bytes(doc.to_bytes())
    assert doc2.has_annotation('DEP')