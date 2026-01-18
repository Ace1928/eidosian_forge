import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_shapes(self, paths):
    """Assert that shapes are transformed correctly"""
    for key, path in paths.items():
        if 'eight' in key or 'missing' in key:
            continue
        inference_data = self.get_inference_data(path)
        test_dict = {'posterior': ['x', 'y', 'Z']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert inference_data.posterior['y'].shape == (4, 100)
        assert inference_data.posterior['x'].shape == (4, 100, 3)
        assert inference_data.posterior['Z'].shape == (4, 100, 4, 6)
        dims = ['chain', 'draw']
        y_mean_true = 0
        y_mean = inference_data.posterior['y'].mean(dim=dims)
        assert np.isclose(y_mean, y_mean_true, atol=0.1)
        x_mean_true = np.array([1, 2, 3])
        x_mean = inference_data.posterior['x'].mean(dim=dims)
        assert np.isclose(x_mean, x_mean_true, atol=0.1).all()
        Z_mean_true = np.array([1, 2, 3, 4])
        Z_mean = inference_data.posterior['Z'].mean(dim=dims).mean(axis=1)
        assert np.isclose(Z_mean, Z_mean_true, atol=0.7).all()
        assert 'comments' in inference_data.posterior.attrs