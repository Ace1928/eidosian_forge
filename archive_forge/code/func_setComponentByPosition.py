import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
def setComponentByPosition(self, idx, value=noValue, verifyConstraints=True, matchTags=True, matchConstraints=True):
    """Assign |ASN.1| type component by position.

        Equivalent to Python sequence item assignment operation (e.g. `[]`).

        Parameters
        ----------
        idx: :class:`int`
            Component index (zero-based). Must either refer to existing
            component or to N+1 component. In the latter case a new component
            type gets instantiated (if *componentType* is set, or given ASN.1
            object is taken otherwise) and appended to the |ASN.1| sequence.

        Keyword Args
        ------------
        value: :class:`object` or :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
            A Python value to initialize |ASN.1| component with (if *componentType* is set)
            or ASN.1 value object to assign to |ASN.1| component. Once a new value is
            set to *idx* component, previous value is dropped.
            If `value` is not given, schema object will be set as a component.

        verifyConstraints : :class:`bool`
            If :obj:`False`, skip constraints validation

        matchTags: :class:`bool`
            If :obj:`False`, skip component tags matching

        matchConstraints: :class:`bool`
            If :obj:`False`, skip component constraints matching

        Returns
        -------
        self
        """
    oldIdx = self._currentIdx
    Set.setComponentByPosition(self, idx, value, verifyConstraints, matchTags, matchConstraints)
    self._currentIdx = idx
    if oldIdx is not None and oldIdx != idx:
        self._componentValues[oldIdx] = noValue
    return self