import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
def test_lowercase_augmenter(nlp, doc):
    augmenter = create_lower_casing_augmenter(level=1.0)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in eg.reference.ents] == ents
    for ref_ent, orig_ent in zip(eg.reference.ents, doc.ents):
        assert ref_ent.text == orig_ent.text.lower()
    assert [t.ent_iob for t in doc] == [t.ent_iob for t in eg.reference]
    assert [t.pos_ for t in eg.reference] == [t.pos_ for t in doc]
    words = ['A', 'B', 'CCC.']
    doc = Doc(nlp.vocab, words=words)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    assert [t.text for t in eg.reference] == [t.lower() for t in words]
    assert [t.text for t in eg.predicted] == [t.text for t in nlp.make_doc(doc.text.lower())]