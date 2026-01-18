import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_observed_data2(self, observed_data_paths):
    """Read Rdump/JSON, check shapes are correct

        One variable as str
        """
    for data_idx in (1, 2):
        path = observed_data_paths[data_idx]
        inference_data = self.get_inference_data(posterior=None, observed_data=path, observed_data_var='x')
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 1
        assert inference_data.observed_data['x'].shape == (1,)