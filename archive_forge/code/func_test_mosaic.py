from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
@pytest.mark.matplotlib
def test_mosaic(close_figures):
    affairs = datasets.fair.load_pandas()
    datas = affairs.exog
    datas['cheated'] = affairs.endog > 0
    datas = datas.sort_values(['rate_marriage', 'religious'])
    num_to_desc = {1: 'awful', 2: 'bad', 3: 'intermediate', 4: 'good', 5: 'wonderful'}
    datas['rate_marriage'] = datas['rate_marriage'].map(num_to_desc)
    num_to_faith = {1: 'non religious', 2: 'poorly religious', 3: 'religious', 4: 'very religious'}
    datas['religious'] = datas['religious'].map(num_to_faith)
    num_to_cheat = {False: 'faithful', True: 'cheated'}
    datas['cheated'] = datas['cheated'].map(num_to_cheat)
    _, ax = plt.subplots(2, 2)
    mosaic(datas, ['rate_marriage', 'cheated'], ax=ax[0, 0], title='by marriage happiness')
    mosaic(datas, ['religious', 'cheated'], ax=ax[0, 1], title='by religiosity')
    mosaic(datas, ['rate_marriage', 'religious', 'cheated'], ax=ax[1, 0], title='by both', labelizer=lambda k: '')
    ax[1, 0].set_xlabel('marriage rating')
    ax[1, 0].set_ylabel('religion status')
    mosaic(datas, ['religious', 'rate_marriage'], ax=ax[1, 1], title='inter-dependence', axes_label=False)
    plt.suptitle('extramarital affairs (plot 3 of 4)')