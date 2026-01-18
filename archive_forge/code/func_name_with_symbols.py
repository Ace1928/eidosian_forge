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
@property
def name_with_symbols(self):
    """
        Similar to :attr:`.name`, but like :attr:`.name` returns also the
        symbols, for example assuming the following function definition::

            def foo(param=0):
                pass

        completing ``foo(`` would give a ``Completion`` which
        ``name_with_symbols`` would be "param=".

        """
    return self._complete(False)