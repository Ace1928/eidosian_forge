import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
@pytest.mark.issue(3880)
def test_issue3880():
    """Test that `nlp.pipe()` works when an empty string ends the batch.

    Fixed in v7.0.5 of Thinc.
    """
    texts = ['hello', 'world', '', '']
    nlp = English()
    nlp.add_pipe('parser').add_label('dep')
    nlp.add_pipe('ner').add_label('PERSON')
    nlp.add_pipe('tagger').add_label('NN')
    nlp.initialize()
    for doc in nlp.pipe(texts):
        pass