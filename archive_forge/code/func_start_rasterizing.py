import numpy as np
from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing
def start_rasterizing(self):
    """
        Enter "raster" mode.  All subsequent drawing commands (until
        `stop_rasterizing` is called) will be drawn with the raster backend.
        """
    self.figure.dpi = self.dpi
    if self._bbox_inches_restore:
        r = process_figure_for_rasterizing(self.figure, self._bbox_inches_restore)
        self._bbox_inches_restore = r
    self._raster_renderer = self._raster_renderer_class(self._width * self.dpi, self._height * self.dpi, self.dpi)
    self._renderer = self._raster_renderer