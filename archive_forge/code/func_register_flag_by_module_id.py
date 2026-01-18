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
def register_flag_by_module_id(self, module_id, flag):
    """Records the module that defines a specific flag.

    Args:
      module_id: int, the ID of the Python module.
      flag: Flag, the Flag instance that is key to the module.
    """
    flags_by_module_id = self.flags_by_module_id_dict()
    flags_by_module_id.setdefault(module_id, []).append(flag)