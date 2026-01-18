import numpy as np
from scipy import special, stats
def kernel_pdf_bs(x, sample, bw):
    return stats.fatiguelife.pdf(sample, bw, scale=x)