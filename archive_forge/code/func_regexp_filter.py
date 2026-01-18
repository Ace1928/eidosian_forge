import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@param.parameterized.bothmethod
def regexp_filter(self_or_cls, pattern):
    """
        Builds a parameter filter using the supplied pattern (may be a
        general Python regular expression)
        """

    def inner_filter(name, p):
        name_match = re.search(pattern, name)
        if name_match is not None:
            return True
        if p.doc is not None:
            doc_match = re.search(pattern, p.doc)
            if doc_match is not None:
                return True
        return False
    return inner_filter