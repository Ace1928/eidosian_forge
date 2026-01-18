import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def push_write(self, s):
    """
        Places new content on the stack.
        """
    self._writes.append(s)