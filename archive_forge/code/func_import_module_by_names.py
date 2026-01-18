import os
from pathlib import Path
from parso.python import tree
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.file_io import FolderIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference import sys_path
from jedi.inference import helpers
from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.utils import unite
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.typeshed import import_module_decorator, \
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.plugins import plugin_manager
def import_module_by_names(inference_state, import_names, sys_path=None, module_context=None, prefer_stubs=True):
    if sys_path is None:
        sys_path = inference_state.get_sys_path()
    str_import_names = tuple((i.value if isinstance(i, tree.Name) else i for i in import_names))
    value_set = [None]
    for i, name in enumerate(import_names):
        value_set = ValueSet.from_sets([import_module(inference_state, str_import_names[:i + 1], parent_module_value, sys_path, prefer_stubs=prefer_stubs) for parent_module_value in value_set])
        if not value_set:
            message = 'No module named ' + '.'.join(str_import_names)
            if module_context is not None:
                _add_error(module_context, name, message)
            else:
                debug.warning(message)
            return NO_VALUES
    return value_set