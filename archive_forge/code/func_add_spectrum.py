from __future__ import annotations
import importlib
import matplotlib.pyplot as plt
from pymatgen.util.plotting import pretty_plot
def add_spectrum(self, label, spectrum, color=None):
    """
        Adds a Spectrum for plotting.

        Args:
            label (str): Label for the Spectrum. Must be unique.
            spectrum: Spectrum object
            color (str): This is passed on to matplotlib. E.g., "k--" indicates
                a dashed black line. If None, a color will be chosen based on
                the default color cycle.
        """
    for attribute in 'xy':
        if not hasattr(spectrum, attribute):
            raise ValueError(f'spectrum of type={type(spectrum)} missing required attribute={attribute!r}')
    self._spectra[label] = spectrum
    self.colors.append(color or self.colors_cycle[len(self._spectra) % len(self.colors_cycle)])