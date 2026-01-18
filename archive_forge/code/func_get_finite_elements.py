import logging
import bisect
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.set import SortedScalarSet
from pyomo.core.base.component import ModelComponentFactory
def get_finite_elements(self):
    """Returns the finite element points

        If the :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>` has been
        discretizaed using a collocation scheme, this method will return a
        list of the finite element discretization points but not the
        collocation points within each finite element. If the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>` has not been
        discretized or a finite difference discretization was used,
        this method returns a list of all the discretization points in the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>`.

        Returns
        -------
        `list` of `floats`
        """
    return self._fe