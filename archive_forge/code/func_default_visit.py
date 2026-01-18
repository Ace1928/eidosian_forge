import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def default_visit(self, node, *args, **kwargs):
    raise NotImplementedError(node['type'])