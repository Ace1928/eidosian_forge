from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.enable_strict_mode')
def enable_strict_mode():
    """If called, enables strict mode for all behaviors.

  Used to switch all deprecation warnings to raise errors instead.
  """
    global STRICT_MODE
    STRICT_MODE = True