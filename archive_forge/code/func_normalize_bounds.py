import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def normalize_bounds(self):
    """Normalizes this NumericRange.

        This returns a normalized range by reversing lb and ub if the
        NumericRange step is less than zero.  If lb and ub are
        reversed, then closed is updated to reflect that change.

        Returns
        -------
        lb, ub, closed

        """
    if self.step >= 0:
        return (self.start, self.end, self.closed)
    else:
        return (self.end, self.start, (self.closed[1], self.closed[0]))