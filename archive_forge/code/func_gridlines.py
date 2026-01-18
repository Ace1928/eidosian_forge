import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def gridlines(self, crs=None, draw_labels=False, xlocs=None, ylocs=None, dms=False, x_inline=None, y_inline=None, auto_inline=True, xformatter=None, yformatter=None, xlim=None, ylim=None, rotate_labels=None, xlabel_style=None, ylabel_style=None, labels_bbox_style=None, xpadding=5, ypadding=5, offset_angle=25, auto_update=None, formatter_kwargs=None, **kwargs):
    """
        Automatically add gridlines to the axes, in the given coordinate
        system, at draw time.

        Parameters
        ----------
        crs: optional
            The :class:`cartopy._crs.CRS` defining the coordinate system in
            which gridlines are drawn.
            Defaults to :class:`cartopy.crs.PlateCarree`.
        draw_labels: optional
            Toggle whether to draw labels. For finer control, attributes of
            :class:`Gridliner` may be modified individually. Defaults to False.

            - string: "x" or "y" to only draw labels of the respective
              coordinate in the CRS.
            - list: Can contain the side identifiers and/or coordinate
              types to select which ones to draw.
              For all labels one would use
              `["x", "y", "top", "bottom", "left", "right", "geo"]`.
            - dict: The keys are the side identifiers
              ("top", "bottom", "left", "right") and the values are the
              coordinates ("x", "y"); this way you can precisely
              decide what kind of label to draw and where.
              For x labels on the bottom and y labels on the right you
              could pass in `{"bottom": "x", "left": "y"}`.

            Note that, by default, x and y labels are not drawn on left/right
            and top/bottom edges respectively unless explicitly requested.

        xlocs: optional
            An iterable of gridline locations or a
            :class:`matplotlib.ticker.Locator` instance which will be
            used to determine the locations of the gridlines in the
            x-coordinate of the given CRS. Defaults to None, which
            implies automatic locating of the gridlines.
        ylocs: optional
            An iterable of gridline locations or a
            :class:`matplotlib.ticker.Locator` instance which will be
            used to determine the locations of the gridlines in the
            y-coordinate of the given CRS. Defaults to None, which
            implies automatic locating of the gridlines.
        dms: bool
            When default longitude and latitude locators and formatters are
            used, ticks are able to stop on minutes and seconds if minutes is
            set to True, and not fraction of degrees. This keyword is passed
            to :class:`~cartopy.mpl.gridliner.Gridliner` and has no effect
            if xlocs and ylocs are explicitly set.
        x_inline: optional
            Toggle whether the x labels drawn should be inline.
        y_inline: optional
            Toggle whether the y labels drawn should be inline.
        auto_inline: optional
            Set x_inline and y_inline automatically based on projection
        xformatter: optional
            A :class:`matplotlib.ticker.Formatter` instance to format labels
            for x-coordinate gridlines. It defaults to None, which implies the
            use of a :class:`cartopy.mpl.ticker.LongitudeFormatter` initiated
            with the ``dms`` argument, if the crs is of
            :class:`~cartopy.crs.PlateCarree` type.
        yformatter: optional
            A :class:`matplotlib.ticker.Formatter` instance to format labels
            for y-coordinate gridlines. It defaults to None, which implies the
            use of a :class:`cartopy.mpl.ticker.LatitudeFormatter` initiated
            with the ``dms`` argument, if the crs is of
            :class:`~cartopy.crs.PlateCarree` type.
        xlim: optional
            Set a limit for the gridlines so that they do not go all the
            way to the edge of the boundary. xlim can be a single number or
            a (min, max) tuple. If a single number, the limits will be
            (-xlim, +xlim).
        ylim: optional
            Set a limit for the gridlines so that they do not go all the
            way to the edge of the boundary. ylim can be a single number or
            a (min, max) tuple. If a single number, the limits will be
            (-ylim, +ylim).
        rotate_labels: optional, bool, str
            Allow the rotation of non-inline labels.

            - False: Do not rotate the labels.
            - True: Rotate the labels parallel to the gridlines.
            - None: no rotation except for some projections (default).
            - A float: Rotate labels by this value in degrees.

        xlabel_style: dict
            A dictionary passed through to ``ax.text`` on x label creation
            for styling of the text labels.
        ylabel_style: dict
            A dictionary passed through to ``ax.text`` on y label creation
            for styling of the text labels.
        labels_bbox_style: dict
            bbox style for all text labels.
        xpadding: float
            Padding for x labels. If negative, the labels are
            drawn inside the map.
        ypadding: float
            Padding for y labels. If negative, the labels are
            drawn inside the map.
        offset_angle: float
            Difference of angle in degrees from 90 to define when
            a label must be flipped to be more readable.
            For example, a value of 10 makes a vertical top label to be
            flipped only at 100 degrees.
        auto_update: bool, default=True
            Whether to update the gridlines and labels when the plot is
            refreshed.

            .. deprecated:: 0.23
               In future the gridlines and labels will always be updated.

        formatter_kwargs: dict, optional
            Options passed to the default formatters.
            See :class:`~cartopy.mpl.ticker.LongitudeFormatter` and
            :class:`~cartopy.mpl.ticker.LatitudeFormatter`

        Keyword Parameters
        ------------------
        **kwargs: dict
            All other keywords control line properties.  These are passed
            through to :class:`matplotlib.collections.Collection`.

        Returns
        -------
        gridliner
            A :class:`cartopy.mpl.gridliner.Gridliner` instance.

        Notes
        -----
        The "x" and "y" for locations and inline settings do not necessarily
        correspond to X and Y, but to the first and second coordinates of the
        specified CRS. For the common case of PlateCarree gridlines, these
        correspond to longitudes and latitudes. Depending on the projection
        used for the map, meridians and parallels can cross both the X axis and
        the Y axis.
        """
    if crs is None:
        crs = ccrs.PlateCarree(globe=self.projection.globe)
    from cartopy.mpl.gridliner import Gridliner
    gl = Gridliner(self, crs=crs, draw_labels=draw_labels, xlocator=xlocs, ylocator=ylocs, collection_kwargs=kwargs, dms=dms, x_inline=x_inline, y_inline=y_inline, auto_inline=auto_inline, xformatter=xformatter, yformatter=yformatter, xlim=xlim, ylim=ylim, rotate_labels=rotate_labels, xlabel_style=xlabel_style, ylabel_style=ylabel_style, labels_bbox_style=labels_bbox_style, xpadding=xpadding, ypadding=ypadding, offset_angle=offset_angle, auto_update=auto_update, formatter_kwargs=formatter_kwargs)
    self.add_artist(gl)
    return gl