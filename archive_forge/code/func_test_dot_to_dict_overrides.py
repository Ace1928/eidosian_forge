import ctypes
import os
from pathlib import Path
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_torch_mps_gpu
from spacy import prefer_gpu, require_cpu, require_gpu, util
from spacy.about import __version__ as spacy_version
from spacy.lang.en import English
from spacy.lang.nl import Dutch
from spacy.language import DEFAULT_CONFIG_PATH
from spacy.ml._precomputable_affine import (
from spacy.schemas import ConfigSchemaTraining, TokenPattern, TokenPatternSchema
from spacy.training.batchers import minibatch_by_words
from spacy.util import (
from .util import get_random_doc, make_tempdir
from spacy import Language
@pytest.mark.parametrize('dot_notation,expected', [({'token.pos': True, 'token._.xyz': True}, {'token': {'pos': True, '_': {'xyz': True}}}), ({'training.batch_size': 128, 'training.optimizer.learn_rate': 0.01}, {'training': {'batch_size': 128, 'optimizer': {'learn_rate': 0.01}}}), ({'attribute_ruler.scorer': {'@scorers': 'spacy.tagger_scorer.v1'}}, {'attribute_ruler': {'scorer': {'@scorers': 'spacy.tagger_scorer.v1'}}})])
def test_dot_to_dict_overrides(dot_notation, expected):
    result = util.dot_to_dict(dot_notation)
    assert result == expected
    assert util.dict_to_dot(result, for_overrides=True) == dot_notation