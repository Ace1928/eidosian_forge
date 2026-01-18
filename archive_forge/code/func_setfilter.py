from suds.sax import Namespace
from suds.sax.element import Element
from suds.plugin import DocumentPlugin, DocumentContext
from logging import getLogger
def setfilter(self, filter):
    """
        Set the filter.
        @param filter: A filter to set.
        @type filter: L{TnsFilter}
        """
    self.filter = filter