from tensorflow.python import pywrap_tfe
def register_watched_variable_resolver(resolver):
    """Registers the resolver to be used to get the list of variables to watch.

  Args:
    resolver: callable, takes a Variable and returns a list of Variables that
      shall be watched.
  """
    global _variables_override
    assert _variables_override is default_get_variables
    _variables_override = resolver