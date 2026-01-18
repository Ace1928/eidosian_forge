from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
def striptags(self):
    """Return a copy of the text with all XML/HTML tags removed.
        
        :return: a `Markup` instance with all tags removed
        :rtype: `Markup`
        :see: `genshi.util.striptags`
        """
    return Markup(striptags(self))