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
@pytest.fixture()
def merged_dict():
    return {'ids': [1, 2, 3, 4, 5, 6, 7], 'words': ['Hi', 'there', 'everyone', 'It', 'is', 'just', 'me'], 'spaces': [True, True, True, True, True, True, False], 'tags': ['INTJ', 'ADV', 'PRON', 'PRON', 'AUX', 'ADV', 'PRON'], 'sent_starts': [1, 0, 0, 1, 0, 0, 0]}