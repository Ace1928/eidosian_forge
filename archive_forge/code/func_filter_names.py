import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def filter_names(inference_state, completion_names, stack, like_name, fuzzy, cached_name):
    comp_dct = set()
    if settings.case_insensitive_completion:
        like_name = like_name.lower()
    for name in completion_names:
        string = name.string_name
        if settings.case_insensitive_completion:
            string = string.lower()
        if helpers.match(string, like_name, fuzzy=fuzzy):
            new = classes.Completion(inference_state, name, stack, len(like_name), is_fuzzy=fuzzy, cached_name=cached_name)
            k = (new.name, new.complete)
            if k not in comp_dct:
                comp_dct.add(k)
                tree_name = name.tree_name
                if tree_name is not None:
                    definition = tree_name.get_definition()
                    if definition is not None and definition.type == 'del_stmt':
                        continue
                yield new