from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
def apply_translated_message(self, original_message: str, translated_message: str) -> None:
    for i, (title, docname) in enumerate(self['entries']):
        if title == original_message:
            self['entries'][i] = (translated_message, docname)
    if self.get('rawcaption') == original_message:
        self['caption'] = translated_message