from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_display_compositor_definition(self):
    definition = ' display factory(Image * Image * Image) RGBTEST'
    self.line_magic('compositor', definition)
    compositors = [c for c in Compositor.definitions if c.group == 'RGBTEST']
    self.assertEqual(len(compositors), 1)
    self.assertEqual(compositors[0].group, 'RGBTEST')
    self.assertEqual(compositors[0].mode, 'display')