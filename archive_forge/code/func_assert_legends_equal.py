import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba
from numpy.testing import assert_array_equal
def assert_legends_equal(leg1, leg2):
    assert leg1.get_title().get_text() == leg2.get_title().get_text()
    for t1, t2 in zip(leg1.get_texts(), leg2.get_texts()):
        assert t1.get_text() == t2.get_text()
    assert_artists_equal(leg1.get_patches(), leg2.get_patches())
    assert_artists_equal(leg1.get_lines(), leg2.get_lines())