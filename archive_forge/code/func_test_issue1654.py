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
@pytest.mark.issue(1654)
def test_issue1654():
    nlp = Language(Vocab())
    assert not nlp.pipeline

    @Language.component('component')
    def component(doc):
        return doc
    nlp.add_pipe('component', name='1')
    nlp.add_pipe('component', name='2', after='1')
    nlp.add_pipe('component', name='3', after='2')
    assert nlp.pipe_names == ['1', '2', '3']
    nlp2 = Language(Vocab())
    assert not nlp2.pipeline
    nlp2.add_pipe('component', name='3')
    nlp2.add_pipe('component', name='2', before='3')
    nlp2.add_pipe('component', name='1', before='2')
    assert nlp2.pipe_names == ['1', '2', '3']