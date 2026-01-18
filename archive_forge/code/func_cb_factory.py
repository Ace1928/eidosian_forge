from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
def cb_factory(self):

    def cb(intern_x0, steptol=1e-08, ftol=1e-12, maxiter=100):
        if isinstance(ftol, float):
            ftol = ftol * np.ones_like(intern_x0)
        self.steptol = steptol
        self.ftol = ftol
        cur_x = np.array(intern_x0)
        iter_idx = 0
        success = False
        self.history_x.append(cur_x.copy())
        while iter_idx < maxiter:
            f = np.asarray(self.f(cur_x))
            self.history_f.append(f)
            rms_f = rms(f)
            self.history_rms_f.append(rms_f)
            self.history_dx.append(self.step(cur_x, iter_idx, maxiter))
            cur_x += self.history_dx[-1]
            self.history_x.append(cur_x.copy())
            iter_idx += 1
            if np.all(np.abs(f) < ftol):
                success = True
                break
        return {'x': cur_x, 'success': success, 'nit': iter_idx, 'nfev': self.nfev, 'njev': self.njev}
    self.alloc()
    return cb