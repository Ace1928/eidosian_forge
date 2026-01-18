import string
from ..sage_helper import _within_sage, sage_method
def fox_derivative(word, phi, var):
    """
    Given a homorphism phi of a group pi, computes
    phi( d word / d var), i.e. the image of the fox derivative
    of the word with respect to var.
    """
    R, phi_ims, fox_ders = setup_fox_derivative(word, phi, var)
    D = 0
    for w in reverse_word(word):
        D = fox_ders[w] + phi_ims[w] * D
    return D