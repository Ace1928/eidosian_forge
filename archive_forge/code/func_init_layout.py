import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def init_layout(self, key, element, ranges, is_geo=False):
    el = element.traverse(lambda x: x, [Element])
    el = el[0] if el else element
    layout = dict(title=self._format_title(key, separator=' '), plot_bgcolor=self.bgcolor, uirevision=True)
    if not self.responsive:
        layout['width'] = self.width
        layout['height'] = self.height
    extent = self.get_extents(element, ranges)
    if len(extent) == 4:
        l, b, r, t = extent
    else:
        l, b, z0, r, t, z1 = extent
    dims = self._get_axis_dims(el)
    if len(dims) > 2:
        xdim, ydim, zdim = dims
    else:
        xdim, ydim = dims
        zdim = None
    xlabel, ylabel, zlabel = self._get_axis_labels(dims)
    if self.invert_axes:
        if is_geo:
            raise ValueError('The invert_axes parameter is not supported on Tiles elements with the plotly backend')
        xlabel, ylabel = (ylabel, xlabel)
        ydim, xdim = (xdim, ydim)
        l, b, r, t = (b, l, t, r)
    if 'x' not in self.labelled:
        xlabel = ''
    if 'y' not in self.labelled:
        ylabel = ''
    if 'z' not in self.labelled:
        zlabel = ''
    xaxis = {}
    if xdim and (not is_geo):
        try:
            if any(np.isnan([r, l])):
                r, l = (0, 1)
        except TypeError:
            pass
        xrange = [r, l] if self.invert_xaxis else [l, r]
        xaxis = dict(range=xrange, title=xlabel)
        if self.logx:
            xaxis['type'] = 'log'
            xaxis['range'] = np.log10(xaxis['range'])
        self._get_ticks(xaxis, self.xticks)
        if self.projection != '3d' and self.xaxis:
            xaxis['automargin'] = False
            if isinstance(xdim, (list, tuple)):
                dim_str = '-'.join([f'{d.name}^{d.label}^{d.unit}' for d in xdim])
            else:
                dim_str = f'{xdim.name}^{xdim.label}^{xdim.unit}'
            xaxis['_dim'] = dim_str
            if 'bare' in self.xaxis:
                xaxis['ticks'] = ''
                xaxis['showticklabels'] = False
                xaxis['title'] = ''
            if 'top' in self.xaxis:
                xaxis['side'] = 'top'
            else:
                xaxis['side'] = 'bottom'
    yaxis = {}
    if ydim and (not is_geo):
        try:
            if any(np.isnan([b, t])):
                b, t = (0, 1)
        except TypeError:
            pass
        yrange = [t, b] if self.invert_yaxis else [b, t]
        yaxis = dict(range=yrange, title=ylabel)
        if self.logy:
            yaxis['type'] = 'log'
            yaxis['range'] = np.log10(yaxis['range'])
        self._get_ticks(yaxis, self.yticks)
        if self.projection != '3d' and self.yaxis:
            yaxis['automargin'] = False
            if isinstance(ydim, (list, tuple)):
                dim_str = '-'.join([f'{d.name}^{d.label}^{d.unit}' for d in ydim])
            else:
                dim_str = f'{ydim.name}^{ydim.label}^{ydim.unit}'
            yaxis['_dim'] = (dim_str,)
            if 'bare' in self.yaxis:
                yaxis['ticks'] = ''
                yaxis['showticklabels'] = False
                yaxis['title'] = ''
            if 'right' in self.yaxis:
                yaxis['side'] = 'right'
            else:
                yaxis['side'] = 'left'
    if is_geo:
        mapbox = {}
        if all((np.isfinite(v) for v in (l, b, r, t))):
            x_center = (l + r) / 2.0
            y_center = (b + t) / 2.0
            lons, lats = Tiles.easting_northing_to_lon_lat([x_center], [y_center])
            mapbox['center'] = dict(lat=lats[0], lon=lons[0])
            margin_left, margin_bottom, margin_right, margin_top = self.margins
            viewport_width = self.width - margin_left - margin_right
            viewport_height = self.height - margin_top - margin_bottom
            mapbox_tile_size = 512
            max_delta = 2 * np.pi * 6378137
            x_delta = r - l
            y_delta = t - b
            with np.errstate(divide='ignore'):
                max_x_zoom = np.log2(max_delta / x_delta) - np.log2(mapbox_tile_size / viewport_width)
                max_y_zoom = np.log2(max_delta / y_delta) - np.log2(mapbox_tile_size / viewport_height)
            mapbox['zoom'] = min(max_x_zoom, max_y_zoom)
        layout['mapbox'] = mapbox
    if isinstance(self.projection, str) and self.projection == '3d':
        scene = dict(xaxis=xaxis, yaxis=yaxis)
        if zdim:
            zrange = [z1, z0] if self.invert_zaxis else [z0, z1]
            zaxis = dict(range=zrange, title=zlabel)
            if self.logz:
                zaxis['type'] = 'log'
            self._get_ticks(zaxis, self.zticks)
            scene['zaxis'] = zaxis
        if self.aspect == 'cube':
            scene['aspectmode'] = 'cube'
        else:
            scene['aspectmode'] = 'manual'
            scene['aspectratio'] = self.aspect
        layout['scene'] = scene
    else:
        l, b, r, t = self.margins
        layout['margin'] = dict(l=l, r=r, b=b, t=t, pad=4)
        if not is_geo:
            layout['xaxis'] = xaxis
            layout['yaxis'] = yaxis
    return layout