from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
def extract_original_messages(self) -> List[str]:
    messages: List[str] = []
    messages.extend(self.get('rawentries', []))
    if 'rawcaption' in self:
        messages.append(self['rawcaption'])
    return messages