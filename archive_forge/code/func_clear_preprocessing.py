import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def clear_preprocessing(self):
    self.__init__(import_rename=self.import_rename, upgrade_compat_v1_import=self.upgrade_compat_v1_import)