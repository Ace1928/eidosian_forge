from ...sage_helper import _within_sage, sage_method
@sage_method
def is_imaginary_part_bounded(z, v):
    """
    Check that the imaginary part of z is in (-v, v).
    """
    imag = z.imag()
    return -v < imag and imag < v