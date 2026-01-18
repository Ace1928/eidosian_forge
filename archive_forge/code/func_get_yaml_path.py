import logging
import os
def get_yaml_path(builtin_name, runtime=''):
    """Returns the full path to a yaml file by giving the builtin module's name.

  Args:
    builtin_name: single word name of builtin handler
    runtime: name of the runtime

  Raises:
    ValueError: if handler does not exist in expected directory

  Returns:
    the absolute path to a valid builtin handler include.yaml file
  """
    if _handler_dir is None:
        set_builtins_dir(DEFAULT_DIR)
    available_builtins = set(_available_builtins)
    if runtime == 'python27':
        available_builtins = available_builtins - BUILTINS_NOT_AVAIABLE_IN_PYTHON27
    if builtin_name not in available_builtins:
        raise InvalidBuiltinName('%s is not the name of a valid builtin.\nAvailable handlers are: %s' % (builtin_name, ', '.join(sorted(available_builtins))))
    return _get_yaml_path(builtin_name, runtime)