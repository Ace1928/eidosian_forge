from numpy import sqrt
from .gradient_descent import GradientDescentOptimizer
@property
def sm(self):
    """Returns estimated second moments of gradient"""
    return None if self.accumulation is None else self.accumulation['sm']