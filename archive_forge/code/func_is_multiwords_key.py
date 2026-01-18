import re
from typing import Any, Dict, List
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
def is_multiwords_key(self, parts: List[str]) -> bool:
    if len(parts) >= 3 and parts[1].strip() == '':
        name = (parts[0].lower(), parts[2].lower())
        if name in self.multiwords_keys:
            return True
        else:
            return False
    else:
        return False