import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.sppatch import expect_v2
from .distparams import distcont
def check_cont_basic():
    for distname, distargs in distcont[:]:
        distfn = getattr(stats, distname)
        m, v, s, k = distfn.stats(*distargs, **dict(moments='mvsk'))
        st = np.array([m, v, s, k])
        mask = np.isfinite(st)
        if mask.sum() < 4:
            distnonfinite.append(distname)
        print(distname)
        expect = distfn.expect
        expect = lambda *args, **kwds: expect_v2(distfn, *args, **kwds)
        special_kwds = specialcases.get(distname, {})
        mnc0 = expect(mom_nc0, args=distargs, **special_kwds)
        mnc1 = expect(args=distargs, **special_kwds)
        mnc2 = expect(mom_nc2, args=distargs, **special_kwds)
        mnc3 = expect(mom_nc3, args=distargs, **special_kwds)
        mnc4 = expect(mom_nc4, args=distargs, **special_kwds)
        mnc1_lc = expect(args=distargs, loc=1, scale=2, **special_kwds)
        try:
            me, ve, se, ke = mnc2mvsk((mnc1, mnc2, mnc3, mnc4))
        except:
            print('exception', mnc1, mnc2, mnc3, mnc4, st)
            me, ve, se, ke = [np.nan] * 4
            if mask.size > 0:
                distex.append(distname)
        em = np.array([me, ve, se, ke])
        diff = st[mask] - em[mask]
        print(diff, mnc1_lc - (1 + 2 * mnc1))
        if np.size(diff) > 0 and np.max(np.abs(diff)) > 0.001:
            distlow.append(distname)
        else:
            distok.append(distname)
        res[distname] = [mnc0, st, em, diff, mnc1_lc]