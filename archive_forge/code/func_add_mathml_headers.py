from sympy.utilities.pkgdata import get_resource
from sympy.utilities.decorator import doctest_depends_on
def add_mathml_headers(s):
    return '<math xmlns:mml="http://www.w3.org/1998/Math/MathML"\n      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n      xsi:schemaLocation="http://www.w3.org/1998/Math/MathML\n        http://www.w3.org/Math/XMLSchema/mathml2/mathml2.xsd">' + s + '</math>'