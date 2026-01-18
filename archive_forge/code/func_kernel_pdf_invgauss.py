import numpy as np
from scipy import special, stats
def kernel_pdf_invgauss(x, sample, bw):
    m = x
    lam = 1 / bw
    return stats.invgauss.pdf(sample, m / lam, scale=lam)