import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def plot_errors(self, concs, init_concs, varied_data, varied, axes=None, compositions=True, Q=True, subplot_kwargs=None):
    if axes is None:
        import matplotlib.pyplot as plt
        if subplot_kwargs is None:
            subplot_kwargs = dict(xscale='log')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=subplot_kwargs)
    varied_idx = self.as_substance_index(varied)
    ls, c = ('- -- : -.'.split(), 'krgbcmy')
    all_inits = np.tile(self.as_per_substance_array(init_concs), (len(varied_data), 1))
    all_inits[:, varied_idx] = varied_data
    if compositions:
        cmp_nrs, m1, m2 = self.composition_conservation(concs, all_inits)
        for cidx, (cmp_nr, a1, a2) in enumerate(zip(cmp_nrs, m1, m2)):
            axes[0].plot(concs[:, varied_idx], a1 - a2, label='Comp ' + str(cmp_nr), ls=ls[cidx % len(ls)], c=c[cidx % len(c)])
            axes[1].plot(concs[:, varied_idx], (a1 - a2) / np.abs(a2), label='Comp ' + str(cmp_nr), ls=ls[cidx % len(ls)], c=c[cidx % len(c)])
    if Q:
        qs = self.equilibrium_quotients(concs)
        ks = [rxn.param for rxn in self.rxns]
        for idx, (q, k) in enumerate(zip(qs, ks)):
            axes[0].plot(concs[:, varied_idx], q - k, label='K R:' + str(idx), ls=ls[(idx + cidx) % len(ls)], c=c[(idx + cidx) % len(c)])
            axes[1].plot(concs[:, varied_idx], (q - k) / k, label='K R:' + str(idx), ls=ls[(idx + cidx) % len(ls)], c=c[(idx + cidx) % len(c)])
    from pyneqsys.plotting import mpl_outside_legend
    mpl_outside_legend(axes[0])
    mpl_outside_legend(axes[1])
    axes[0].set_title('Absolute errors')
    axes[1].set_title('Relative errors')