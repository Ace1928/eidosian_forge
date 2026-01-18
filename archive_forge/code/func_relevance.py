from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
def relevance(error):
    validator = error.validator
    return (-len(error.path), validator not in weak, validator in strong)