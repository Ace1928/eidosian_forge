from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.scanner import Scanner, RoundTripScanner, ScannerError  # NOQA
from ruamel.yaml.scanner import BlankLineComment
from ruamel.yaml.comments import C_PRE, C_POST, C_SPLIT_ON_FIRST_BLANK
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.tag import Tag
def select_tag_transform(self, tag: Tag) -> None:
    if tag is None:
        return
    tag.select_transform(True)