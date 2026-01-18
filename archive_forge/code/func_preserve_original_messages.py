from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
def preserve_original_messages(self) -> None:
    rawentries = self.setdefault('rawentries', [])
    for title, _docname in self['entries']:
        if title:
            rawentries.append(title)
    if self.get('caption'):
        self['rawcaption'] = self['caption']