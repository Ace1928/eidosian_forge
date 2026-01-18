import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('blob_args', [(ValueError, ['a', 'b'], ['prior']), (ValueError, ['too', 'many', 'names'], None), (SyntaxError, ['a', 'b'], ['posterior', 'observed_data'])])
def test_bad_blobs(self, data, blob_args):
    error, names, groups = blob_args
    with pytest.raises(error):
        from_emcee(data.obj, blob_names=names, blob_groups=groups)