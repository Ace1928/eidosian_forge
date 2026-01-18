import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_observed_data1(self, observed_data_paths):
    """Read Rdump/JSON, check shapes are correct

        All variables
        """
    for data_idx in (1, 2):
        path = observed_data_paths[data_idx]
        inference_data = self.get_inference_data(posterior=None, observed_data=path)
        assert hasattr(inference_data, 'observed_data')
        assert len(inference_data.observed_data.data_vars) == 3
        assert inference_data.observed_data['x'].shape == (1,)
        assert inference_data.observed_data['x'][0] == 1
        assert inference_data.observed_data['y'].shape == (3,)
        assert inference_data.observed_data['Z'].shape == (4, 5)