from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def str_unprocessed(self) -> Any:
    return ''.join((f'  {ind:2} {x.info()}\n' for ind, x in self.comments.items() if x.used == ' '))