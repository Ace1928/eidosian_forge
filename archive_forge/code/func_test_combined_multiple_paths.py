from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_combined_multiple_paths(self):
    line = "Image Curve {+framewise} [fig_inches=(3, 3) title='foo bar'] (c='b') Layout [string='foo'] Overlay"
    expected = {'Image': {'norm': Options(framewise=True, axiswise=False), 'plot': Options(title='foo bar', fig_inches=(3, 3)), 'style': Options(c='b')}, 'Curve': {'norm': Options(framewise=True, axiswise=False), 'plot': Options(title='foo bar', fig_inches=(3, 3)), 'style': Options(c='b')}, 'Layout': {'plot': Options(string='foo')}, 'Overlay': {}}
    self.assertEqual(OptsSpec.parse(line), expected)