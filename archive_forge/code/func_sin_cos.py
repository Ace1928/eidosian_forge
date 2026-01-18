from math import exp, sin, cos
def sin_cos(data):
    """Sine cosine benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [0, 6]^2`
       * - Function
         - :math:`f(\\mathbf{x}) = 6\\sin(x_1)\\cos(x_2)`
    """
    6 * sin(data[0]) * cos(data[1])