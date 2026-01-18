import sys
import numpy as np
from bokeh.palettes import all_palettes
from param import concrete_descendents
from ...core import (
from ...core.options import Cycle, Options, Palette
from ...element import (
from ..plot import PlotSelector
from ..util import fire
from .annotation import (
from .callbacks import Callback  # noqa (API import)
from .chart import (
from .element import ElementPlot, OverlayPlot
from .geometry import RectanglesPlot, SegmentPlot
from .graphs import ChordPlot, GraphPlot, NodePlot, TriMeshPlot
from .heatmap import HeatMapPlot, RadialHeatMapPlot
from .hex_tiles import HexTilesPlot
from .links import LinkCallback  # noqa (API import)
from .path import ContourPlot, PathPlot, PolygonPlot
from .plot import AdjointLayoutPlot, GridPlot, LayoutPlot
from .raster import HSVPlot, ImageStackPlot, QuadMeshPlot, RasterPlot, RGBPlot
from .renderer import BokehRenderer
from .sankey import SankeyPlot
from .stats import BivariatePlot, BoxWhiskerPlot, DistributionPlot, ViolinPlot
from .tabular import TablePlot
from .tiles import TilePlot
from .util import bokeh_version  # noqa (API import)
def colormap_generator(palette):
    epsilon = sys.float_info.epsilon * 10
    return lambda value: palette[int(value * (len(palette) - 1) + epsilon)]