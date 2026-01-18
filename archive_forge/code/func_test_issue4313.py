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
@pytest.mark.issue(4313)
def test_issue4313():
    """This should not crash or exit with some strange error code"""
    beam_width = 16
    beam_density = 0.0001
    nlp = English()
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    ner = nlp.add_pipe('beam_ner', config=config)
    ner.add_label('SOME_LABEL')
    nlp.initialize()
    doc = nlp('What do you think about Apple ?')
    assert len(ner.labels) == 1
    assert 'SOME_LABEL' in ner.labels
    apple_ent = Span(doc, 5, 6, label='MY_ORG')
    doc.ents = list(doc.ents) + [apple_ent]
    docs = [doc]
    ner.beam_parse(docs, drop=0.0, beam_width=beam_width, beam_density=beam_density)
    assert len(ner.labels) == 2
    assert 'MY_ORG' in ner.labels