from math import exp, sin, cos
def kotanchek(data):
    """Kotanchek benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [-1, 7]^2`
       * - Function
         - :math:`f(\\mathbf{x}) = \\\\frac{e^{-(x_1 - 1)^2}}{3.2 + (x_2 - 2.5)^2}`
    """
    return exp(-(data[0] - 1) ** 2) / (3.2 + (data[1] - 2.5) ** 2)