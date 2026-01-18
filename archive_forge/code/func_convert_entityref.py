import html.entities
import re
from .sgml import *
@staticmethod
def convert_entityref(name):
    """
        :type name: str
        :rtype: str
        """
    return '&%s;' % name