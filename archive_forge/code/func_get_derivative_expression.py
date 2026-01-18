import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
def get_derivative_expression(self):
    """
        Returns the current discretization expression for this derivative or
        creates an access function to its :py:class:`Var` the first time
        this method is called. The expression gets built up as the
        discretization transformations are sequentially applied to each
        :py:class:`ContinuousSet` in the model.
        """
    try:
        return self._expr
    except:
        self._expr = create_access_function(self._sVar)
        return self._expr