import time
import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context, get_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('backend', ['loky', 'threading', 'multiprocessing'])
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    with config_context(working_memory=123):
        results = Parallel(n_jobs=n_jobs, backend=backend)((delayed(get_working_memory)() for _ in range(2)))
    assert_array_equal(results, [123] * 2)