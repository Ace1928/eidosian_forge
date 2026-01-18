import string
from ..sage_helper import _within_sage, sage_method
def clean_CC(z, error):
    CC = z.parent()
    return CC(clean_RR(z.real(), error), clean_RR(z.imag(), error)) if not CC.is_exact() else z