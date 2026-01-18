from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def get_environments_figure(self, isite, plot_type=None, title='Coordination numbers', max_dist=2.0, colormap=None, figsize=None, strategy=None):
    """
        Plotting of the coordination environments of a given site for all the distfactor/angfactor regions. The
        chemical environments with the lowest continuous symmetry measure is shown for each distfactor/angfactor
        region as the value for the color of that distfactor/angfactor region (using a colormap).

        Args:
            isite: Index of the site for which the plot has to be done.
            plot_type: How to plot the coordinations.
            title: Title for the figure.
            max_dist: Maximum distance to be plotted when the plotting of the distance is set to 'initial_normalized'
                or 'initial_real' (Warning: this is not the same meaning in both cases! In the first case, the
                closest atom lies at a "normalized" distance of 1.0 so that 2.0 means refers to this normalized
                distance while in the second case, the real distance is used).
            colormap: Color map to be used for the continuous symmetry measure.
            figsize: Size of the figure.
            strategy: Whether to plot information about one of the Chemenv Strategies.

        Returns:
            tuple[plt.Figure, plt.Axes]: matplotlib figure and axes representing the environments.
        """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if plot_type is None:
        plot_type = {'distance_parameter': ('initial_normalized', None), 'angle_parameter': ('initial_normalized_inverted', None)}
    clr_map = cm.jet if colormap is None else colormap
    clr_min = 0.0
    clr_max = 10.0
    norm = Normalize(vmin=clr_min, vmax=clr_max)
    scalarmap = cm.ScalarMappable(norm=norm, cmap=clr_map)
    dist_limits = [1.0, max_dist]
    ang_limits = [0.0, 1.0]
    if plot_type['distance_parameter'][0] == 'one_minus_inverse_alpha_power_n':
        if plot_type['distance_parameter'][1] is None:
            exponent = 3
        else:
            exponent = plot_type['distance_parameter'][1]['exponent']
        xlabel = f'Distance parameter : $1.0-\\frac{{1.0}}{{\\alpha^{{{exponent}}}}}$'

        def dp_func(dp):
            return 1.0 - 1.0 / np.power(dp, exponent)
    elif plot_type['distance_parameter'][0] == 'initial_normalized':
        xlabel = 'Distance parameter : $\\alpha$'

        def dp_func(dp):
            return dp
    else:
        raise ValueError(f'Wrong value for distance parameter plot type "{plot_type['distance_parameter'][0]}"')
    if plot_type['angle_parameter'][0] == 'one_minus_gamma':
        ylabel = 'Angle parameter : $1.0-\\gamma$'

        def ap_func(ap):
            return 1.0 - ap
    elif plot_type['angle_parameter'][0] in ['initial_normalized_inverted', 'initial_normalized']:
        ylabel = 'Angle parameter : $\\gamma$'

        def ap_func(ap):
            return ap
    else:
        raise ValueError(f'Wrong value for angle parameter plot type "{plot_type['angle_parameter'][0]}"')
    dist_limits = [dp_func(dp) for dp in dist_limits]
    ang_limits = [ap_func(ap) for ap in ang_limits]
    for cn, cn_nb_sets in self.neighbors_sets[isite].items():
        for inb_set, nb_set in enumerate(cn_nb_sets):
            nb_set_surface_pts = nb_set.voronoi_grid_surface_points()
            if nb_set_surface_pts is None:
                continue
            ce = self.ce_list[isite][cn][inb_set]
            if ce is None:
                color = 'w'
                inv_color = 'k'
                text = f'{cn}'
            else:
                mingeom = ce.minimum_geometry()
                if mingeom is not None:
                    mp_symbol = mingeom[0]
                    csm = mingeom[1]['symmetry_measure']
                    color = scalarmap.to_rgba(csm)
                    inv_color = [1.0 - color[0], 1.0 - color[1], 1.0 - color[2], 1.0]
                    text = f'{mp_symbol}'
                else:
                    color = 'w'
                    inv_color = 'k'
                    text = f'{cn}'
            nb_set_surface_pts = [(dp_func(pt[0]), ap_func(pt[1])) for pt in nb_set_surface_pts]
            polygon = Polygon(nb_set_surface_pts, closed=True, edgecolor='k', facecolor=color, linewidth=1.2)
            ax.add_patch(polygon)
            ipt = len(nb_set_surface_pts) / 2
            if ipt != int(ipt):
                raise RuntimeError('Uneven number of surface points')
            ipt = int(ipt)
            patch_center = ((nb_set_surface_pts[0][0] + min(nb_set_surface_pts[ipt][0], dist_limits[1])) / 2, (nb_set_surface_pts[0][1] + nb_set_surface_pts[ipt][1]) / 2)
            if np.abs(nb_set_surface_pts[-1][1] - nb_set_surface_pts[-2][1]) > 0.06 and np.abs(min(nb_set_surface_pts[-1][0], dist_limits[1]) - nb_set_surface_pts[0][0]) > 0.125:
                xytext = ((min(nb_set_surface_pts[-1][0], dist_limits[1]) + nb_set_surface_pts[0][0]) / 2, (nb_set_surface_pts[-1][1] + nb_set_surface_pts[-2][1]) / 2)
                ax.annotate(text, xy=xytext, ha='center', va='center', color=inv_color, fontsize='x-small')
            elif np.abs(nb_set_surface_pts[ipt][1] - nb_set_surface_pts[0][1]) > 0.1 and np.abs(min(nb_set_surface_pts[ipt][0], dist_limits[1]) - nb_set_surface_pts[0][0]) > 0.125:
                xytext = patch_center
                ax.annotate(text, xy=xytext, ha='center', va='center', color=inv_color, fontsize='x-small')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    dist_limits.sort()
    ang_limits.sort()
    ax.set_xlim(dist_limits)
    ax.set_ylim(ang_limits)
    if strategy is not None:
        try:
            strategy.add_strategy_visualization_to_subplot(subplot=ax)
        except Exception:
            pass
    if plot_type['angle_parameter'][0] == 'initial_normalized_inverted':
        ax.axes.invert_yaxis()
    scalarmap.set_array([clr_min, clr_max])
    cb = fig.colorbar(scalarmap, ax=ax, extend='max')
    cb.set_label('Continuous symmetry measure')
    return (fig, ax)