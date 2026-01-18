import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_deep_clone_map_select_redim(self):
    annotations = Text(0, 0, 'A') + Arrow(0, 0) + HLine(0) + VLine(0)
    selected = annotations.select(x=(0, 5))
    redimmed = selected.redim(x='z')
    relabelled = redimmed.relabel(label='foo', depth=5)
    mapped = relabelled.map(lambda x: x.clone(group='bar'), Annotation)
    kwargs = dict(label='foo', group='bar', extents=(0, None, 5, None), kdims=['z', 'y'])
    self.assertEqual(mapped.Text.I, Text(0, 0, 'A', **kwargs))
    self.assertEqual(mapped.Arrow.I, Arrow(0, 0, **kwargs))
    self.assertEqual(mapped.HLine.I, HLine(0, **kwargs))
    self.assertEqual(mapped.VLine.I, VLine(0, **kwargs))