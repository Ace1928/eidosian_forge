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
@memoize_method
def defined_names(self):
    """
        List sub-definitions (e.g., methods in class).

        :rtype: list of :class:`Name`
        """
    defs = self._name.infer()
    return sorted(unite((defined_names(self._inference_state, d) for d in defs)), key=lambda s: s._name.start_pos or (0, 0))