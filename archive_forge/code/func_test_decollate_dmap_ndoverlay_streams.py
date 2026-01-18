from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_decollate_dmap_ndoverlay_streams(self):
    self.perform_decollate_dmap_container_streams(NdOverlay)