from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
@property
def child_text_separator(self):
    if self.get('is_multiline'):
        return ' '
    else:
        return super().child_text_separator