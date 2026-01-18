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
@method(function('transform', nargs=1, sequence_types=('map(*)', 'map(*)')))
def evaluate_transform_function(self, context=None):
    if self.context is not None:
        context = self.context
    options = self.get_argument(context, required=True, cls=XPathMap)
    for k, v in options.items(context):
        if k == 'xslt-version':
            if not isinstance(v, (int, float, Decimal)):
                raise self.error('FOXT0002')
    raise self.error('FOXT0004')