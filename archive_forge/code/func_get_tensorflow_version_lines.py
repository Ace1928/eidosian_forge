import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def get_tensorflow_version_lines(include_dependency_versions=False):
    """Generate RichTextLines with TensorFlow version info.

  Args:
    include_dependency_versions: Include the version of TensorFlow's key
      dependencies, such as numpy.

  Returns:
    A formatted, multi-line `RichTextLines` object.
  """
    lines = ['TensorFlow version: %s' % pywrap_tf_session.__version__]
    lines.append('')
    if include_dependency_versions:
        lines.append('Dependency version(s):')
        lines.append('  numpy: %s' % np.__version__)
        lines.append('')
    return RichTextLines(lines)