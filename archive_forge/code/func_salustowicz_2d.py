from math import exp, sin, cos
def salustowicz_2d(data):
    """Salustowicz benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [0, 7]^2`
       * - Function
         - :math:`f(\\mathbf{x}) = e^{-x_1} x_1^3 \\cos(x_1) \\sin(x_1) (\\cos(x_1) \\sin^2(x_1) - 1) (x_2 -5)`
    """
    return exp(-data[0]) * data[0] ** 3 * cos(data[0]) * sin(data[0]) * (cos(data[0]) * sin(data[0]) ** 2 - 1) * (data[1] - 5)