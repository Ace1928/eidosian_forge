import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
def create_spongebob_augmenter(randomize: bool=False):

    def augment(nlp, example):
        text = example.text
        if randomize:
            ch = [c.lower() if random.random() < 0.5 else c.upper() for c in text]
        else:
            ch = [c.lower() if i % 2 else c.upper() for i, c in enumerate(text)]
        example_dict = example.to_dict()
        doc = nlp.make_doc(''.join(ch))
        example_dict['token_annotation']['ORTH'] = [t.text for t in doc]
        yield example
        yield example.from_dict(doc, example_dict)
    return augment