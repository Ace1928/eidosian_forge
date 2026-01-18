import json
import typing
import warnings
from io import BytesIO
from typing import (
from warnings import warn
import jmespath
from lxml import etree, html
from packaging.version import Version
from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten
def create_root_node(text: str, parser_cls: Type[_ParserType], base_url: Optional[str]=None, huge_tree: bool=LXML_SUPPORTS_HUGE_TREE, body: bytes=b'', encoding: str='utf8') -> etree._Element:
    """Create root node for text using given parser class."""
    if not text:
        body = body.replace(b'\x00', b'').strip()
    else:
        body = text.strip().replace('\x00', '').encode(encoding) or b'<html/>'
    if huge_tree and LXML_SUPPORTS_HUGE_TREE:
        parser = parser_cls(recover=True, encoding=encoding, huge_tree=True)
        root = etree.fromstring(body, parser=parser, base_url=base_url)
    else:
        parser = parser_cls(recover=True, encoding=encoding)
        root = etree.fromstring(body, parser=parser, base_url=base_url)
        for error in parser.error_log:
            if 'use XML_PARSE_HUGE option' in error.message:
                warnings.warn(f'Input data is too big. Upgrade to lxml {lxml_huge_tree_version} or later for huge_tree support.')
    if root is None:
        root = etree.fromstring(b'<html/>', parser=parser, base_url=base_url)
    return root