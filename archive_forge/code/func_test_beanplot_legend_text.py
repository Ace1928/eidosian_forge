import numpy as np
import pytest
from statsmodels.datasets import anes96
from statsmodels.graphics.boxplots import beanplot, violinplot
@pytest.mark.matplotlib
def test_beanplot_legend_text(age_and_labels, close_figures):
    age, labels = age_and_labels
    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels, plot_opts={'bean_legend_text': 'text'})