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
@pytest.mark.issue(5458)
def test_issue5458():
    words = ['In', 'an', 'era', 'where', 'markets', 'have', 'brought', 'prosperity', 'and', 'empowerment', '.']
    vocab = Vocab(strings=words)
    deps = ['ROOT', 'det', 'pobj', 'advmod', 'nsubj', 'aux', 'relcl', 'dobj', 'cc', 'conj', 'punct']
    pos = ['ADP', 'DET', 'NOUN', 'ADV', 'NOUN', 'AUX', 'VERB', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT']
    heads = [0, 2, 0, 9, 6, 6, 2, 6, 7, 7, 0]
    en_doc = Doc(vocab, words=words, pos=pos, heads=heads, deps=deps)
    en_doc.noun_chunks_iterator = noun_chunks
    nlp = English()
    merge_nps = nlp.create_pipe('merge_noun_chunks')
    merge_nps(en_doc)