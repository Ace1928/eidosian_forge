from wasabi import Printer
from ...errors import Errors
from ...tokens import Doc, Span
from ...training import iob_to_biluo, tags_to_entities
from ...util import minibatch
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def iob_to_docs(input_data, n_sents=10, no_print=False, *args, **kwargs):
    """
    Convert IOB files with one sentence per line and tags separated with '|'
    into Doc objects so they can be saved. IOB and IOB2 are accepted.

    Sample formats:

    I|O like|O London|I-GPE and|O New|B-GPE York|I-GPE City|I-GPE .|O
    I|O like|O London|B-GPE and|O New|B-GPE York|I-GPE City|I-GPE .|O
    I|PRP|O like|VBP|O London|NNP|I-GPE and|CC|O New|NNP|B-GPE York|NNP|I-GPE City|NNP|I-GPE .|.|O
    I|PRP|O like|VBP|O London|NNP|B-GPE and|CC|O New|NNP|B-GPE York|NNP|I-GPE City|NNP|I-GPE .|.|O
    """
    vocab = Vocab()
    msg = Printer(no_print=no_print)
    if n_sents > 0:
        n_sents_info(msg, n_sents)
    yield from read_iob(input_data.split('\n'), vocab, n_sents)