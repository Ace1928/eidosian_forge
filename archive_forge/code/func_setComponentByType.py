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
def setComponentByType(self, tagSet, value=noValue, verifyConstraints=True, matchTags=True, matchConstraints=True, innerFlag=False):
    """Assign |ASN.1| type component by ASN.1 tag.

        Parameters
        ----------
        tagSet : :py:class:`~pyasn1.type.tag.TagSet`
            Object representing ASN.1 tags to identify one of
            |ASN.1| object component

        Keyword Args
        ------------
        value: :class:`object` or :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
            A Python value to initialize |ASN.1| component with (if *componentType* is set)
            or ASN.1 value object to assign to |ASN.1| component.
            If `value` is not given, schema object will be set as a component.

        verifyConstraints : :class:`bool`
            If :obj:`False`, skip constraints validation

        matchTags: :class:`bool`
            If :obj:`False`, skip component tags matching

        matchConstraints: :class:`bool`
            If :obj:`False`, skip component constraints matching

        innerFlag: :class:`bool`
            If :obj:`True`, search for matching *tagSet* recursively.

        Returns
        -------
        self
        """
    idx = self.componentType.getPositionByType(tagSet)
    if innerFlag:
        componentType = self.componentType.getTypeByPosition(idx)
        if componentType.tagSet:
            return self.setComponentByPosition(idx, value, verifyConstraints, matchTags, matchConstraints)
        else:
            componentType = self.getComponentByPosition(idx)
            return componentType.setComponentByType(tagSet, value, verifyConstraints, matchTags, matchConstraints, innerFlag=innerFlag)
    else:
        return self.setComponentByPosition(idx, value, verifyConstraints, matchTags, matchConstraints)