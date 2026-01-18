import json
import locale
import math
import pathlib
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import product
from urllib.request import urlopen
from urllib.parse import urlsplit
from ..datatypes import AnyAtomicType, AbstractBinary, AbstractDateTime, \
from ..exceptions import ElementPathTypeError
from ..helpers import WHITESPACES_PATTERN, is_xml_codepoint, \
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XML_BASE
from ..etree import etree_iter_strings, is_etree_element
from ..collations import CollationManager
from ..compare import get_key_function, same_key
from ..tree_builders import get_node_tree
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode
from ..xpath_tokens import XPathFunction, XPathMap, XPathArray
from ..xpath_context import XPathSchemaContext
from ..validators import validate_json_to_xml
from ._xpath31_operators import XPath31Parser
@method(function('collation-key', nargs=(1, 2), sequence_types=('xs:string', 'xs:string', 'xs:base64Binary')))
def evaluate_collation_key_function(self, context=None):
    if self.context is not None:
        context = self.context
    key = self.get_argument(context, required=True, cls=str)
    if len(self) > 1:
        collation = self.get_argument(context, index=1, required=True, cls=str)
    else:
        collation = self.parser.default_collation
    try:
        with CollationManager(collation, self) as manager:
            base64_key = Base64Binary.encoder(manager.strxfrm(key).encode())
            return Base64Binary(base64_key, ordered=True)
    except locale.Error:
        raise self.error('FOCH0004')