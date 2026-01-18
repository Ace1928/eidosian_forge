import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(3959)
def test_issue3959():
    """Ensure that a modified pos attribute is serialized correctly."""
    nlp = English()
    doc = nlp('displaCy uses JavaScript, SVG and CSS to show you how computers understand language')
    assert doc[0].pos_ == ''
    doc[0].pos_ = 'NOUN'
    assert doc[0].pos_ == 'NOUN'
    with make_tempdir() as tmp_dir:
        file_path = tmp_dir / 'my_doc'
        doc.to_disk(file_path)
        doc2 = nlp('')
        doc2.from_disk(file_path)
        assert doc2[0].pos_ == 'NOUN'