import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def peek_write(self):
    """
        Returns the last content written to the document without
        removing it from the stack.
        """
    return self._writes[-1]