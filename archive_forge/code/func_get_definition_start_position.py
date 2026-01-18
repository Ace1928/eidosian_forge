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
def get_definition_start_position(self):
    """
        The (row, column) of the start of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
    if self._name.tree_name is None:
        return None
    definition = self._name.tree_name.get_definition()
    if definition is None:
        return self._name.start_pos
    return definition.start_pos