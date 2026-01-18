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
def test_PrecomputableAffine(nO=4, nI=5, nF=3, nP=2):
    model = PrecomputableAffine(nO=nO, nI=nI, nF=nF, nP=nP).initialize()
    assert model.get_param('W').shape == (nF, nO, nP, nI)
    tensor = model.ops.alloc((10, nI))
    Y, get_dX = model.begin_update(tensor)
    assert Y.shape == (tensor.shape[0] + 1, nF, nO, nP)
    dY = model.ops.alloc((15, nO, nP))
    ids = model.ops.alloc((15, nF))
    ids[1, 2] = -1
    dY[1] = 1
    assert not model.has_grad('pad')
    d_pad = _backprop_precomputable_affine_padding(model, dY, ids)
    assert d_pad[0, 2, 0, 0] == 1.0
    ids.fill(0.0)
    dY.fill(0.0)
    dY[0] = 0
    ids[1, 2] = 0
    ids[1, 1] = -1
    ids[1, 0] = -1
    dY[1] = 1
    ids[2, 0] = -1
    dY[2] = 5
    d_pad = _backprop_precomputable_affine_padding(model, dY, ids)
    assert d_pad[0, 0, 0, 0] == 6
    assert d_pad[0, 1, 0, 0] == 1
    assert d_pad[0, 2, 0, 0] == 0