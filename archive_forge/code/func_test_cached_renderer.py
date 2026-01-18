import os
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
@pytest.mark.backend('macosx')
def test_cached_renderer():
    fig = plt.figure(1)
    fig.canvas.draw()
    assert fig.canvas.get_renderer()._renderer is not None
    fig = plt.figure(2)
    fig.draw_without_rendering()
    assert fig.canvas.get_renderer()._renderer is not None