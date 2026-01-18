import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
    opts = {} if isinstance(self, HvOverlayPlot) else {'source': source}
    fig = super().initialize_plot(ranges, plot, plots, **opts)
    if self.geographic and self.show_bounds and (not self.overlaid):
        from . import GeoShapePlot
        shape = Shape(self.projection.boundary, crs=self.projection).options(fill_alpha=0)
        shapeplot = GeoShapePlot(shape, projection=self.projection, overlaid=True, renderer=self.renderer)
        shapeplot.geographic = False
        shapeplot.initialize_plot(plot=fig)
    return fig