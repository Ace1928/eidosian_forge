import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
@needs_emcee3
@pytest.mark.parametrize('test_args', arg_list)
def test_inference_data_reader(self, test_args):
    kwargs, test_dict = test_args
    kwargs = {k: i for k, i in kwargs.items() if k not in ('arg_names', 'arg_groups')}
    inference_data = self.get_inference_data_reader(**kwargs)
    test_dict.pop('observed_data')
    if 'constant_data' in test_dict:
        test_dict.pop('constant_data')
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails