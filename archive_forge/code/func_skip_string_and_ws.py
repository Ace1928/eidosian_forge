import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def skip_string_and_ws(self, string: str) -> bool:
    if self.skip_string(string):
        self.skip_ws()
        return True
    return False