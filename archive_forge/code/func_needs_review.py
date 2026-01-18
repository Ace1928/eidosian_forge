from numpy.testing import dec
from nibabel.data import DataError
def needs_review(msg):
    """Skip a test that needs further review.

    Parameters
    ----------
    msg : string
        msg regarding the review that needs to be done
    """

    def skip_func(func):
        return dec.skipif(True, msg)(func)
    return skip_func