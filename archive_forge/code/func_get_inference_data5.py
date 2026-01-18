import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
def get_inference_data5(self, data, eight_schools_params):
    """multiple vars as lists."""
    return from_cmdstanpy(posterior=data.obj, posterior_predictive=None, prior=data.obj, prior_predictive=None, log_likelihood='log_lik', observed_data={'y': eight_schools_params['y']}, coords=None, dims=None, dtypes=data.model.code())