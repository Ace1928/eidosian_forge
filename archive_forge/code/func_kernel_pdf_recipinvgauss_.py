import numpy as np
from scipy import special, stats
def kernel_pdf_recipinvgauss_(x, sample, bw):
    """Reciprocal inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """
    pdf = 1 / np.sqrt(2 * np.pi * bw * sample) * np.exp(-(x - bw) / (2 * bw) * sample / (x - bw) - 2 + (x - bw) / sample)
    return pdf