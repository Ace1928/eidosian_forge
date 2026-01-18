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
def test_gold_constructor():
    """Test that the Example constructor works fine"""
    nlp = English()
    doc = nlp('This is a sentence')
    example = Example.from_dict(doc, {'cats': {'cat1': 1.0, 'cat2': 0.0}})
    assert example.get_aligned('ORTH', as_string=True) == ['This', 'is', 'a', 'sentence']
    assert example.reference.cats['cat1']
    assert not example.reference.cats['cat2']