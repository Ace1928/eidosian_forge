from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_combined_multiple_paths_merge(self):
    line = "Image Curve [fig_inches=(3, 3)] (c='b') Image (s=3)"
    expected = {'Image': {'plot': Options(fig_inches=(3, 3)), 'style': Options(c='b', s=3)}, 'Curve': {'plot': Options(fig_inches=(3, 3)), 'style': Options(c='b')}}
    self.assertEqual(OptsSpec.parse(line), expected)