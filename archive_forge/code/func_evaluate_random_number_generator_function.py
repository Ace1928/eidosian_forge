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
@method(function('random-number-generator', nargs=(0, 1), sequence_types=('xs:anyAtomicType?', 'map(xs:string, item())')))
def evaluate_random_number_generator_function(self, context=None):
    if self.context is not None:
        context = self.context
    seed = self.get_argument(context, cls=AnyAtomicType)
    if not isinstance(seed, (int, str)):
        seed = str(seed)
    random.seed(seed)

    class Permute(XPathFunction):
        nargs = 1
        sequence_types = ('item()*', 'item()*')

        def __call__(self, *args, **kwargs):
            if not args:
                return []
            try:
                seq = [x for x in args[0]]
            except TypeError:
                return [args[0]]
            else:
                random.shuffle(seq)
                return seq

    class NextRandom(XPathFunction):
        nargs = 0
        sequence_types = ('map(xs:string, item())',)

        def __call__(self, *args, **kwargs):
            items = {'number': random.random(), 'next': NextRandom(self.parser), 'permute': Permute(self.parser)}
            return XPathMap(self.parser, items)
    return NextRandom(self.parser)()