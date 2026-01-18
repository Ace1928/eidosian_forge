import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def get_canonical_import(import_set):
    """Obtain one single import from a set of possible sources of a symbol.

  One symbol might come from multiple places as it is being imported and
  reexported. To simplify API changes, we always use the same import for the
  same module, and give preference based on higher priority and alphabetical
  ordering.

  Args:
    import_set: (set) Imports providing the same symbol. This is a set of tuples
      in the form (import, priority). We want to pick an import with highest
      priority.

  Returns:
    A module name to import
  """
    import_list = sorted(import_set, key=lambda imp_and_priority: (-imp_and_priority[1], imp_and_priority[0]))
    return import_list[0][0]