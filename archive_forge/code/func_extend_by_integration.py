from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def extend_by_integration(self, xend, params=None, odesys=None, autonomous=None, npoints=1, **kwargs):
    odesys = odesys or self.odesys
    if autonomous is None:
        autonomous = odesys.autonomous_interface
    x0 = self.xout[-1]
    nx0 = self.xout.size
    res = odesys.integrate(self.odesys.numpy.linspace((xend - x0) * 0, xend - x0, npoints + 1) if autonomous else self.odesys.numpy.linspace(x0, xend, npoints + 1), self.yout[..., -1, :], params or self.params, **kwargs)
    self.xout = self.odesys.numpy.concatenate((self.xout, res.xout[1:] + (x0 if autonomous else 0 * x0)))
    self.yout = self.odesys.numpy.concatenate((self.yout, res.yout[..., 1:, :]))
    new_info = {k: v for k, v in self.info.items() if not (k.startswith('internal') and odesys is not self.odesys)}
    for k, v in res.info.items():
        if k.startswith('internal'):
            if odesys is self.odesys:
                new_info[k] = self.odesys.numpy.concatenate((new_info[k], v))
            else:
                continue
        elif k == 'success':
            new_info[k] = new_info[k] and v
        elif k.endswith('_xvals'):
            if len(v) == 0:
                continue
            new_info[k] = self.odesys.numpy.concatenate((new_info[k], v + (x0 if autonomous else 0 * x0)))
        elif k.endswith('_indices'):
            new_info[k].extend([itm + nx0 - 1 for itm in v])
        elif isinstance(v, str):
            if isinstance(new_info[k], str):
                new_info[k] = [new_info[k]]
            new_info[k].append(v)
        else:
            new_info[k] += v
    self.info = new_info
    return self