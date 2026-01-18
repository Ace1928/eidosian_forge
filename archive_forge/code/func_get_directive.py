from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def get_directive(self, name):
    """Return the directive class for the given name.
        
        :param name: the directive name as used in the template
        :return: the directive class
        :see: `Directive`
        """
    return self._dir_by_name.get(name)