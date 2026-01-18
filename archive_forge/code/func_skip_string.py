import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def skip_string(self, string: str) -> bool:
    strlen = len(string)
    if self.definition[self.pos:self.pos + strlen] == string:
        self.pos += strlen
        return True
    return False