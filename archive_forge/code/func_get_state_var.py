import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
def get_state_var(self):
    """Return the :py:class:`Var` that is being differentiated.

        Returns
        -------
        :py:class:`Var<pyomo.environ.Var>`
        """
    return self._sVar