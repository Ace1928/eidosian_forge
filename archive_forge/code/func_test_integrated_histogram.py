import numpy as np
import pytest
import matplotlib.pyplot as plt
from cirq.vis import integrated_histogram
@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('data', [range(10), {f'key_{i}': i for i in range(10)}])
def test_integrated_histogram(data):
    ax = integrated_histogram(data, title='Test Plot', axis_label='Y Axis Label', color='r', label='line label', cdf_on_x=True, show_zero=True)
    assert ax.get_title() == 'Test Plot'
    assert ax.get_ylabel() == 'Y Axis Label'
    assert len(ax.get_lines()) == 2
    for line in ax.get_lines():
        assert line.get_color() == 'r'