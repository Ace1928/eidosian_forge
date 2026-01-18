import sys
from pathlib import Path
import parso
from parso.python import tree
from jedi.parser_utils import get_executable_nodes
from jedi import debug
from jedi import settings
from jedi import cache
from jedi.file_io import KnownContentFileIO
from jedi.api import classes
from jedi.api import interpreter
from jedi.api import helpers
from jedi.api.helpers import validate_line_column
from jedi.api.completion import Completion, search_in_module
from jedi.api.keywords import KeywordName
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import get_default_project, Project
from jedi.api.errors import parso_to_jedi_errors
from jedi.api import refactoring
from jedi.api.refactoring.extract import extract_function, extract_variable
from jedi.inference import InferenceState
from jedi.inference import imports
from jedi.inference.references import find_references
from jedi.inference.arguments import try_iter_content
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.sys_path import transform_path_to_dotted
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value import ModuleValue
from jedi.inference.base_value import ValueSet
from jedi.inference.value.iterable import unpack_tuple_to_dict
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.gradual.utils import load_proper_stub_module
from jedi.inference.utils import to_list
def set_debug_function(func_cb=debug.print_to_stdout, warnings=True, notices=True, speed=True):
    """
    Define a callback debug function to get all the debug messages.

    If you don't specify any arguments, debug messages will be printed to stdout.

    :param func_cb: The callback function for debug messages.
    """
    debug.debug_function = func_cb
    debug.enable_warning = warnings
    debug.enable_notice = notices
    debug.enable_speed = speed