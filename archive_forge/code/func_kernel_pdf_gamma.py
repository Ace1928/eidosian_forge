import numpy as np
from scipy import special, stats
def kernel_pdf_gamma(x, sample, bw):
    pdfi = stats.gamma.pdf(sample, x / bw + 1, scale=bw)
    return pdfi