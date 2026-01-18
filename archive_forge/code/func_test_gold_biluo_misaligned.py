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
def test_gold_biluo_misaligned(en_vocab, en_tokenizer):
    words = ['Mr and Mrs', 'Smith', 'flew', 'to', 'San Francisco', 'Valley', '.']
    spaces = [True, True, True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = 'Mr and Mrs Smith flew to '
    entities = [(len(prefix), len(prefix + 'San Francisco Valley'), 'LOC')]
    gold_words = ['Mr', 'and Mrs Smith', 'flew to', 'San', 'Francisco Valley', '.']
    example = Example.from_dict(doc, {'words': gold_words, 'entities': entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ['O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O']
    entities = [(len('Mr and '), len('Mr and Mrs Smith'), 'PERSON'), (len(prefix), len(prefix + 'San Francisco Valley'), 'LOC')]
    gold_words = ['Mr and', 'Mrs Smith', 'flew to', 'San', 'Francisco Valley', '.']
    example = Example.from_dict(doc, {'words': gold_words, 'entities': entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == [None, None, 'O', 'O', 'B-LOC', 'L-LOC', 'O']