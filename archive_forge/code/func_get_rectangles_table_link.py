import param
import cartopy.crs as ccrs
from holoviews.annotators import (
from holoviews.plotting.links import DataLink, VertexTableLink as hvVertexTableLink
from panel.util import param_name
from .element import Path
from .models.custom_tools import CheckpointTool, RestoreTool, ClearTool
from .links import VertexTableLink, PointTableLink, HvRectanglesTableLink, RectanglesTableLink
from .operation import project
from .streams import PolyVertexDraw, PolyVertexEdit
def get_rectangles_table_link(self, source, target):
    if hasattr(source.callback.inputs[0], 'crs'):
        return RectanglesTableLink(source, target)
    else:
        return HvRectanglesTableLink(source, target)