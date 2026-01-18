from __future__ import unicode_literals
from cmakelang.parse.funs import standard_builtins, standard_modules
from cmakelang.parse.util import CommandSpec
def get_fn_spec():
    """
  Return a dictionary mapping cmake function names to a dictionary containing
  kwarg specifications.
  """
    fn_spec = CommandSpec('<root>')
    for grouping in (standard_builtins, standard_modules):
        for funname, spec in sorted(grouping.FUNSPECS.items()):
            fn_spec.add(funname, **spec)
    return fn_spec