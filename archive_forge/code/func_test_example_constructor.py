import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
def test_example_constructor(en_vocab):
    words = ['I', 'like', 'stuff']
    tags = ['NOUN', 'VERB', 'NOUN']
    tag_ids = [en_vocab.strings.add(tag) for tag in tags]
    predicted = Doc(en_vocab, words=words)
    reference = Doc(en_vocab, words=words)
    reference = reference.from_array('TAG', numpy.array(tag_ids, dtype='uint64'))
    example = Example(predicted, reference)
    tags = example.get_aligned('TAG', as_string=True)
    assert tags == ['NOUN', 'VERB', 'NOUN']