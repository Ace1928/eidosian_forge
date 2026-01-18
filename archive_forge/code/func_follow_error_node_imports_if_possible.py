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
def follow_error_node_imports_if_possible(context, name):
    error_node = tree.search_ancestor(name, 'error_node')
    if error_node is not None:
        start_index = 0
        for index, n in enumerate(error_node.children):
            if n.start_pos > name.start_pos:
                break
            if n == ';':
                start_index = index + 1
        nodes = error_node.children[start_index:]
        first_name = nodes[0].get_first_leaf().value
        if first_name in ('from', 'import'):
            is_import_from = first_name == 'from'
            level, names = helpers.parse_dotted_names(nodes, is_import_from=is_import_from, until_node=name)
            return Importer(context.inference_state, names, context.get_root_context(), level).follow()
    return None