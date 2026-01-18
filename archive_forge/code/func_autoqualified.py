from suds import *
from suds.xsd import *
from suds.sax.element import Element
from suds.sax import Namespace
from logging import getLogger
def autoqualified(self):
    """
        The list of I{auto} qualified attribute values.

        Qualification means to convert values into I{qref}.

        @return: A list of attribute names.
        @rtype: list

        """
    return ['type', 'ref']