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
def test_alignment():
    other_tokens = ['i', 'listened', 'to', 'obama', "'", 's', 'podcasts', '.']
    spacy_tokens = ['i', 'listened', 'to', 'obama', "'s", 'podcasts', '.']
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert list(align.x2y.data) == [0, 1, 2, 3, 4, 4, 5, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 1, 1]
    assert list(align.y2x.data) == [0, 1, 2, 3, 4, 5, 6, 7]