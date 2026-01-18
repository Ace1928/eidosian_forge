import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
def prepare_plot(self, ax=None, emin=-10, emax=5, ylabel=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.figure().add_subplot(111)

    def pretty(kpt):
        if kpt == 'G':
            kpt = '$\\Gamma$'
        elif len(kpt) == 2:
            kpt = kpt[0] + '$_' + kpt[1] + '$'
        return kpt
    self.xcoords, label_xcoords, orig_labels = self.bs.get_labels()
    label_xcoords = list(label_xcoords)
    labels = [pretty(name) for name in orig_labels]
    i = 1
    while i < len(labels):
        if label_xcoords[i - 1] == label_xcoords[i]:
            labels[i - 1] = labels[i - 1] + ',' + labels[i]
            labels.pop(i)
            label_xcoords.pop(i)
        else:
            i += 1
    for x in label_xcoords[1:-1]:
        ax.axvline(x, color='0.5')
    ylabel = ylabel if ylabel is not None else 'energies [eV]'
    ax.set_xticks(label_xcoords)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.axhline(self.bs.reference, color='k', ls=':')
    ax.axis(xmin=0, xmax=self.xcoords[-1], ymin=emin, ymax=emax)
    self.ax = ax
    return ax