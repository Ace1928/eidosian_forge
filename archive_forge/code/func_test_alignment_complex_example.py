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
def test_alignment_complex_example(en_vocab):
    other_tokens = ['i listened to', 'obama', "'", 's', 'podcasts', '.']
    spacy_tokens = ['i', 'listened', 'to', 'obama', "'s", 'podcasts.']
    predicted = Doc(en_vocab, words=other_tokens, spaces=[True, False, False, True, False, False])
    reference = Doc(en_vocab, words=spacy_tokens, spaces=[True, True, True, False, True, False])
    assert predicted.text == "i listened to obama's podcasts."
    assert reference.text == "i listened to obama's podcasts."
    example = Example(predicted, reference)
    align = example.alignment
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1]
    assert list(align.x2y.data) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.data) == [0, 0, 0, 1, 2, 3, 4, 5]