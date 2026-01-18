import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
def get_inference_data_reader(self, **kwargs):
    from emcee import backends
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, '..', 'saved_models')
    filepath = os.path.join(data_directory, 'reader_testfile.h5')
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath)
    reader = backends.HDFBackend(filepath, read_only=True)
    return from_emcee(reader, **kwargs)