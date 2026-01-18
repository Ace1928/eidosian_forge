from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def get_flags_for_module(self, module):
    """Returns the list of flags defined by a module.

    Args:
      module: module|str, the module to get flags from.

    Returns:
      [Flag], a new list of Flag instances.  Caller may update this list as
      desired: none of those changes will affect the internals of this
      FlagValue instance.
    """
    if not isinstance(module, str):
        module = module.__name__
    if module == '__main__':
        module = sys.argv[0]
    return list(self.flags_by_module_dict().get(module, []))