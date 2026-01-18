from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent
def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    """
    Find the optical setup conjugating the object/image waists.

    Parameters
    ==========

    wavelen :
        The wavelength of the beam.
    waist_in and waist_out :
        The waists to be conjugated.
    f :
        The focal distance of the element used in the conjugation.

    Returns
    =======

    a tuple containing (s_in, s_out, f)
    s_in :
        The distance before the optical element.
    s_out :
        The distance after the optical element.
    f :
        The focal distance of the optical element.

    Examples
    ========

    >>> from sympy.physics.optics import conjugate_gauss_beams
    >>> from sympy import symbols, factor
    >>> l, w_i, w_o, f = symbols('l w_i w_o f')

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
              pi**2*w_i**4/(f**2*l**2)))/w_i**2

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
    f
    """
    wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
    m = waist_out / waist_in
    z = waist2rayleigh(waist_in, wavelen)
    if len(kwargs) != 1:
        raise ValueError('The function expects only one named argument')
    elif 'dist' in kwargs:
        raise NotImplementedError(filldedent('\n            Currently only focal length is supported as a parameter'))
    elif 'f' in kwargs:
        f = sympify(kwargs['f'])
        s_in = f * (1 - sqrt(1 / m ** 2 - z ** 2 / f ** 2))
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        raise NotImplementedError(filldedent('\n            Currently only focal length is supported as a parameter'))
    else:
        raise ValueError(filldedent('\n            The functions expects the focal length as a named argument'))
    return (s_in, s_out, f)