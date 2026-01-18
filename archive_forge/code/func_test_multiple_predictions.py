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
def test_multiple_predictions():

    class DummyPipe(TrainablePipe):

        def __init__(self):
            self.model = 'dummy_model'

        def predict(self, docs):
            return ([1, 2, 3], [4, 5, 6])

        def set_annotations(self, docs, scores):
            return docs
    nlp = Language()
    doc = nlp.make_doc('foo')
    dummy_pipe = DummyPipe()
    dummy_pipe(doc)