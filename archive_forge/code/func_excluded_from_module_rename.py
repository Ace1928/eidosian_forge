import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def excluded_from_module_rename(module, import_rename_spec):
    """Check if this module import should not be renamed.

  Args:
    module: (string) module name.
    import_rename_spec: ImportRename instance.

  Returns:
    True if this import should not be renamed according to the
    import_rename_spec.
  """
    for excluded_prefix in import_rename_spec.excluded_prefixes:
        if module.startswith(excluded_prefix):
            return True
    return False