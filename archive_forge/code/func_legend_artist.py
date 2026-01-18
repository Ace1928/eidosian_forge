from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def legend_artist(self, legend, orig_handle, fontsize, handlebox):
    """
        Return the artist that this HandlerBase generates for the given
        original artist/handle.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            The legend for which these legend artists are being created.
        orig_handle : :class:`matplotlib.artist.Artist` or similar
            The object for which these legend artists are being created.
        fontsize : int
            The fontsize in pixels. The artists being created should
            be scaled according to the given fontsize.
        handlebox : `~matplotlib.offsetbox.OffsetBox`
            The box which has been created to hold this legend entry's
            artists. Artists created in the `legend_artist` method must
            be added to this handlebox inside this method.

        """
    xdescent, ydescent, width, height = self.adjust_drawing_area(legend, orig_handle, handlebox.xdescent, handlebox.ydescent, handlebox.width, handlebox.height, fontsize)
    artists = self.create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, handlebox.get_transform())
    for a in artists:
        handlebox.add_artist(a)
    return artists[0]