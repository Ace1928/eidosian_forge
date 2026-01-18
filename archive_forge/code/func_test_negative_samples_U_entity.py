import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_negative_samples_U_entity(tsys, vocab, neg_key):
    """Test that we exclude a 2-word entity correctly using a negative example."""
    tsys.cfg['neg_key'] = neg_key
    doc = Doc(vocab, words=['A'])
    entity_annots = [None]
    example = Example.from_dict(doc, {'entities': entity_annots})
    example.y.spans[neg_key] = [Span(example.y, 0, 1, label='O'), Span(example.y, 0, 1, label='PERSON')]
    act_classes = tsys.get_oracle_sequence(example)
    names = [tsys.get_class_name(act) for act in act_classes]
    assert names
    assert names[0] != 'O'
    assert names[0] != 'U-PERSON'