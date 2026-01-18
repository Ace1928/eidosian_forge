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
@pytest.mark.issue(1506)
def test_issue1506():

    def string_generator():
        for _ in range(10001):
            yield "It's sentence produced by that bug."
        for _ in range(10001):
            yield 'I erase some hbdsaj lemmas.'
        for _ in range(10001):
            yield 'I erase lemmas.'
        for _ in range(10001):
            yield "It's sentence produced by that bug."
        for _ in range(10001):
            yield "It's sentence produced by that bug."
    nlp = English()
    for i, d in enumerate(nlp.pipe(string_generator())):
        if i == 10000 or i == 20000 or i == 30000:
            gc.collect()
        for t in d:
            str(t.lemma_)