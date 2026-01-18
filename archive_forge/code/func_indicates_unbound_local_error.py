import os
from mako.cache import CacheImpl
from mako.cache import register_plugin
from mako.template import Template
from .assertions import eq_
from .config import config
def indicates_unbound_local_error(self, rendered_output, unbound_var):
    var = f'&#39;{unbound_var}&#39;'
    error_msgs = (f'local variable {var} referenced before assignment', f'cannot access local variable {var} where it is not associated')
    return any((msg in rendered_output for msg in error_msgs))