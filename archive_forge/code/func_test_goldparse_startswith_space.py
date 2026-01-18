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
def test_goldparse_startswith_space(en_tokenizer):
    text = ' a'
    doc = en_tokenizer(text)
    gold_words = ['a']
    entities = ['U-DATE']
    deps = ['ROOT']
    heads = [0]
    example = Example.from_dict(doc, {'words': gold_words, 'entities': entities, 'deps': deps, 'heads': heads})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ['O', 'U-DATE']
    assert example.get_aligned('DEP', as_string=True) == [None, 'ROOT']