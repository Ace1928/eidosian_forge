from __future__ import annotations
import abc
import collections
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.plotting import add_fig_kwargs, pretty_plot
def get_plot(self, structure: Structure, two_theta_range: tuple[float, float]=(0, 90), annotate_peaks='compact', ax: plt.Axes=None, with_labels=True, fontsize=16) -> plt.Axes:
    """
        Returns the diffraction plot as a matplotlib Axes.

        Args:
            structure: Input structure
            two_theta_range (tuple[float, float]): Range of two_thetas to calculate in degrees.
                Defaults to (0, 90). Set to None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
            annotate_peaks (str | None): Whether and how to annotate the peaks
                with hkl indices. Default is 'compact', i.e. show short
                version (oriented vertically), e.g. 100. If 'full', show
                long version, e.g. (1, 0, 0). If None, do not show anything.
            ax: matplotlib Axes or None if a new figure should be
                created.
            with_labels: True to add xlabels and ylabels to the plot.
            fontsize: (int) fontsize for peak labels.

        Returns:
            plt.Axes: matplotlib Axes object
        """
    ax = ax or pretty_plot(16, 10)
    xrd = self.get_pattern(structure, two_theta_range=two_theta_range)
    imax = max(xrd.y)
    for two_theta, i, hkls in zip(xrd.x, xrd.y, xrd.hkls):
        if two_theta_range[0] <= two_theta <= two_theta_range[1]:
            hkl_tuples = [hkl['hkl'] for hkl in hkls]
            label = ', '.join(map(str, hkl_tuples))
            ax.plot([two_theta, two_theta], [0, i], color='k', linewidth=3, label=label)
            if annotate_peaks == 'full':
                ax.annotate(label, xy=[two_theta, i], xytext=[two_theta, i], fontsize=fontsize)
            elif annotate_peaks == 'compact':
                if all((all((i < 10 for i in hkl_tuple)) for hkl_tuple in hkl_tuples)):
                    label = ','.join((''.join(map(str, hkl_tuple)) for hkl_tuple in hkl_tuples))
                if i / imax > 0.5:
                    xytext = [-fontsize / 4, 0]
                    ha = 'right'
                    va = 'top'
                else:
                    xytext = [0, 10]
                    ha = 'center'
                    va = 'bottom'
                ax.annotate(label, xy=[two_theta, i], xytext=xytext, textcoords='offset points', va=va, ha=ha, rotation=90, fontsize=fontsize)
    if with_labels:
        ax.set_xlabel('$2\\theta$ ($^\\circ$)')
        ax.set_ylabel('Intensities (scaled)')
    plt.tight_layout()
    return ax