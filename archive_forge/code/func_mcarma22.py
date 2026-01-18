import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arma_mle import Arma
def mcarma22(niter=10, nsample=1000, ar=None, ma=None, sig=0.5):
    """run Monte Carlo for ARMA(2,2)

    DGP parameters currently hard coded
    also sample size `nsample`

    was not a self contained function, used instances from outer scope
      now corrected

    """
    if ar is None:
        ar = [1.0, -0.55, -0.1]
    if ma is None:
        ma = [1.0, 0.3, 0.2]
    results = []
    results_bse = []
    for _ in range(niter):
        y2 = arma_generate_sample(ar, ma, nsample + 1000, sig)[-nsample:]
        y2 -= y2.mean()
        arest2 = Arma(y2)
        rhohat2a, cov_x2a, infodict, mesg, ier = arest2.fit((2, 2))
        results.append(rhohat2a)
        err2a = arest2.geterrors(rhohat2a)
        sige2a = np.sqrt(np.dot(err2a, err2a) / nsample)
        if cov_x2a is not None:
            results_bse.append(sige2a * np.sqrt(np.diag(cov_x2a)))
        else:
            results_bse.append(np.nan + np.zeros_like(rhohat2a))
    return (np.r_[ar[1:], ma[1:]], np.array(results), np.array(results_bse))