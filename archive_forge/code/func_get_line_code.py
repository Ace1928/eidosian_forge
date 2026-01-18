import re
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports
def get_line_code(self, before=0, after=0):
    """
        Returns the line of code where this object was defined.

        :param before: Add n lines before the current line to the output.
        :param after: Add n lines after the current line to the output.

        :return str: Returns the line(s) of code or an empty string if it's a
                     builtin.
        """
    if not self._name.is_value_name:
        return ''
    lines = self._name.get_root_context().code_lines
    if lines is None:
        return ''
    index = self._name.start_pos[0] - 1
    start_index = max(index - before, 0)
    return ''.join(lines[start_index:index + after + 1])