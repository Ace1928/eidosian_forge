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
def test_alignment_different_texts():
    other_tokens = ['she', 'listened', 'to', 'obama', "'s", 'podcasts', '.']
    spacy_tokens = ['i', 'listened', 'to', 'obama', "'s", 'podcasts', '.']
    with pytest.raises(ValueError):
        Alignment.from_strings(other_tokens, spacy_tokens)