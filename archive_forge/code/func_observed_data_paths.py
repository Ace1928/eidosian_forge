import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
@pytest.fixture(scope='class')
def observed_data_paths(self, data_directory):
    observed_data_paths = [os.path.join(data_directory, 'cmdstan/eight_schools.data.R'), os.path.join(data_directory, 'cmdstan/example_stan.data.R'), os.path.join(data_directory, 'cmdstan/example_stan.json')]
    return observed_data_paths