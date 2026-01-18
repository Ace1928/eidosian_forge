from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_cell_opts_style_dynamic(self):
    self.cell("dmap = DynamicMap(lambda X: Curve(np.random.rand(5,2), name='dmap'), kdims=['x']).redim.range(x=(0, 10)).opts({'Curve': dict(linewidth=2, color='black')})")
    self.assertEqual(self.get_object('dmap').id, None)
    self.cell_magic('opts', ' Curve (linewidth=3 alpha=0.5)', 'dmap')
    self.assertEqual(self.get_object('dmap').id, 0)
    assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
    opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'style').options
    self.assertEqual(opts, {'linewidth': 3, 'alpha': 0.5, 'color': 'black'})