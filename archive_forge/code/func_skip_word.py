import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def skip_word(self, word: str) -> bool:
    return self.match(re.compile('\\b%s\\b' % re.escape(word)))