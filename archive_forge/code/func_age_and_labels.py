import numpy as np
import pytest
from statsmodels.datasets import anes96
from statsmodels.graphics.boxplots import beanplot, violinplot
@pytest.fixture(scope='module')
def age_and_labels():
    data = anes96.load_pandas()
    party_ID = np.arange(7)
    labels = ['Strong Democrat', 'Weak Democrat', 'Independent-Democrat', 'Independent-Independent', 'Independent-Republican', 'Weak Republican', 'Strong Republican']
    age = [data.exog['age'][data.endog == id] for id in party_ID]
    age = np.array(age, dtype='object')
    return (age, labels)