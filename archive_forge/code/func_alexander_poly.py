from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def alexander_poly(self, *args, **kwargs):
    """
        Please use the "alexander_polynomial" method instead.
        """
    if 'alexander_poly' not in deprecation_warnings_issued:
        deprecation_warnings_issued.add('alexander_poly')
        print('Deprecation Warning: use "alexander_polynomial" instead of "alexander_poly".')
    return self.alexander_polynomial(*args, **kwargs)