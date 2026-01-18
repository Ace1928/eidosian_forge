from __future__ import annotations
import importlib
import matplotlib.pyplot as plt
from pymatgen.util.plotting import pretty_plot
def add_spectra(self, spectra_dict, key_sort_func=None):
    """
        Add a dictionary of Spectrum, with an optional sorting function for the
        keys.

        Args:
            spectra_dict: dict of {label: Spectrum}
            key_sort_func: function used to sort the dos_dict keys.
        """
    keys = sorted(spectra_dict, key=key_sort_func) if key_sort_func else list(spectra_dict)
    for label in keys:
        self.add_spectrum(label, spectra_dict[label])