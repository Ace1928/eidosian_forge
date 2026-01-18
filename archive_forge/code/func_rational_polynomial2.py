from math import exp, sin, cos
def rational_polynomial2(data):
    """Rational polynomial benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [0, 6]^2`
       * - Function
         - :math:`f(\\mathbf{x}) = \\\\frac{(x_1 - 3)^4 + (x_2 - 3)^3 - (x_2 - 3)}{(x_2 - 2)^4 + 10}`
    """
    return ((data[0] - 3) ** 4 + (data[1] - 3) ** 3 - (data[1] - 3)) / ((data[1] - 2) ** 4 + 10)