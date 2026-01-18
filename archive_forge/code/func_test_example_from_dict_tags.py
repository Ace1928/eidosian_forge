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
def test_example_from_dict_tags(en_vocab):
    words = ['I', 'like', 'stuff']
    tags = ['NOUN', 'VERB', 'NOUN']
    predicted = Doc(en_vocab, words=words)
    example = Example.from_dict(predicted, {'TAGS': tags})
    tags = example.get_aligned('TAG', as_string=True)
    assert tags == ['NOUN', 'VERB', 'NOUN']