from __future__ import absolute_import
import sys
def get_mpl_colormap(self, **kwargs):
    """
        A color map that can be used in matplotlib plots. Requires matplotlib
        to be importable. Keyword arguments are passed to
        `matplotlib.colors.LinearSegmentedColormap.from_list`.

        """
    if not HAVE_MPL:
        raise RuntimeError('matplotlib not available.')
    cmap = LinearSegmentedColormap.from_list(self.name, self.mpl_colors, **kwargs)
    return cmap