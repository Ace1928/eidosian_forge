import functools
import inspect
import sys
import unittest
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.util import tf_inspect
def is_allowlisted(o, check_call_override=True, allow_namedtuple_subclass=False):
    """Checks whether an entity is allowed for use in graph mode.

  Examples of allowed entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
    check_call_override: Reserved for internal use. When set to `False`, it
      disables the rule according to which classes are allowed if their
      __call__ method is allowed.
    allow_namedtuple_subclass: Reserved for internal use. When `True`,
      namedtuple subclasses are not allowed.

  Returns:
    Boolean
  """
    if isinstance(o, functools.partial):
        m = functools
    else:
        m = tf_inspect.getmodule(o)
    if hasattr(m, '__name__'):
        for rule in config.CONVERSION_RULES:
            action = rule.get_action(m)
            if action == config.Action.CONVERT:
                logging.log(2, 'Not allowed: %s: %s', o, rule)
                return False
            elif action == config.Action.DO_NOT_CONVERT:
                logging.log(2, 'Allowlisted: %s: %s', o, rule)
                return True
    if hasattr(o, '__code__') and tf_inspect.isgeneratorfunction(o):
        logging.log(2, 'Allowlisted: %s: generator functions are not converted', o)
        return True
    if check_call_override and (not tf_inspect.isclass(o)) and hasattr(o, '__call__'):
        if type(o) != type(o.__call__) and is_allowlisted(o.__call__):
            logging.log(2, 'Allowlisted: %s: object __call__ allowed', o)
            return True
    owner_class = None
    if tf_inspect.ismethod(o):
        owner_class = inspect_utils.getmethodclass(o)
        if owner_class is tf_method_target.TfMethodTarget:
            owner_class = o.__self__.target_class
        if owner_class is not None:
            if issubclass(owner_class, unittest.TestCase):
                logging.log(2, 'Allowlisted: %s: method of TestCase subclass', o)
                return True
            owner_class = inspect_utils.getdefiningclass(o, owner_class)
            if is_allowlisted(owner_class, check_call_override=False, allow_namedtuple_subclass=True):
                logging.log(2, 'Allowlisted: %s: owner is allowed %s', o, owner_class)
                return True
    if inspect_utils.isnamedtuple(o):
        if allow_namedtuple_subclass:
            if not any((inspect_utils.isnamedtuple(base) for base in o.__bases__)):
                logging.log(2, 'Allowlisted: %s: named tuple', o)
                return True
        else:
            logging.log(2, 'Allowlisted: %s: named tuple or subclass', o)
            return True
    logging.log(2, 'Not allowed: %s: default rule', o)
    return False