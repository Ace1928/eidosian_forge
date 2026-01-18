from ._util import get_backend
from .util.pyutil import defaultnamedtuple, deprecated
from .units import default_units
def get_P_at_T_and_c(self, T, c, **kwargs):
    """Convenience method for calculating concentration

        Calculate the partial pressure for given temperature and concentration


        Parameters
        ----------
        T: float
            Temperature
        P: float
            Pressure
        \\*\\*kwargs:
            Keyword arguments passed on to :meth:`__call__`
        """
    return c / self(T, **kwargs)