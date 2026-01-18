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
@method(function('load-xquery-module', nargs=(1, 2), sequence_types=('xs:string', 'map(*)', 'map(*)')))
def evaluate_load_xquery_module_function(self, context=None):
    if self.context is not None:
        context = self.context
    try:
        module_uri = self.get_argument(context, required=True, cls=str)
    except TypeError:
        raise self.error('FOQM0006')
    if not module_uri:
        raise self.error('FOQM0001')
    if len(self) > 1:
        options = self.get_argument(context, index=1, required=True, cls=XPathMap)
        for k, v in options.items(context):
            if k == 'xquery-version':
                if not isinstance(v, (int, float, Decimal)):
                    raise self.error('FOQM0005')
            elif k == 'location-hints':
                if not isinstance(v, str) or not (isinstance(v, list) and all((isinstance(x, str) for x in v))):
                    raise self.error('FOQM0005')
            elif k == 'context-item':
                if isinstance(v, list) and len(v) > 1:
                    raise self.error('FOQM0005')
            elif k == 'variables' or k == 'vendor-options':
                if not isinstance(v, XPathMap) or any((not isinstance(x, str) for x in v.keys(context))):
                    raise self.error('FOQM0006')
            else:
                raise self.error('FOQM0005')
    raise self.error('FOQM0006')