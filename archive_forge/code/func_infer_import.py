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
@inference_state_method_cache(default=NO_VALUES)
def infer_import(context, tree_name):
    module_context = context.get_root_context()
    from_import_name, import_path, level, values = _prepare_infer_import(module_context, tree_name)
    if values:
        if from_import_name is not None:
            values = values.py__getattribute__(from_import_name, name_context=context, analysis_errors=False)
            if not values:
                path = import_path + (from_import_name,)
                importer = Importer(context.inference_state, path, module_context, level)
                values = importer.follow()
    debug.dbg('after import: %s', values)
    return values