import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
def clean_tag(name):
    """Cleans a tag. Removes illegal characters for instance.

  Args:
    name: The original tag name to be processed.

  Returns:
    The cleaned tag name.
  """
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')
        if new_name != name:
            tf_logging.info('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name