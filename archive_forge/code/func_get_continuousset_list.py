import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
def get_continuousset_list(self):
    """Return the a list of :py:class:`ContinuousSet` components the
        derivative is being taken with respect to.

        Returns
        -------
        `list`
        """
    return self._wrt