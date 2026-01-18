from typing import List
import numpy
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Model, data_validation, get_current_ops
from thinc.types import Array2d, Ragged
from spacy.lang.en import English
from spacy.ml import FeatureExtractor, StaticVectors
from spacy.ml._character_embed import CharacterEmbed
from spacy.tokens import Doc
from spacy.vocab import Vocab
@pytest.mark.parametrize('name', ['tagger', 'tok2vec', 'morphologizer', 'senter'])
def test_components_batching_list(name):
    nlp = English()
    proc = nlp.create_pipe(name)
    util_batch_unbatch_docs_list(proc.model, get_docs(), list_floats)