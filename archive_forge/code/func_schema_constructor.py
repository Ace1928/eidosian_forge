from abc import ABCMeta
import locale
from collections.abc import MutableSequence
from urllib.parse import urlparse
from typing import cast, Any, Callable, ClassVar, Dict, List, \
from ..helpers import upper_camel_case, is_ncname, ordinal
from ..exceptions import ElementPathError, ElementPathTypeError, \
from ..namespaces import NamespacesType, XSD_NAMESPACE, XML_NAMESPACE, \
from ..collations import UNICODE_COLLATION_BASE_URI, UNICODE_CODEPOINT_COLLATION
from ..datatypes import UntypedAtomic, AtomicValueType, QName
from ..xpath_tokens import NargsType, XPathToken, ProxyToken, XPathFunction, XPathConstructor
from ..xpath_context import XPathContext, XPathSchemaContext
from ..sequence_types import is_sequence_type, match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath1 import XPath1Parser
def schema_constructor(self, atomic_type_name: str, bp: int=90) -> Type[XPathFunction]:
    """Registers a token class for a schema atomic type constructor function."""
    if atomic_type_name in (XSD_ANY_ATOMIC_TYPE, XSD_NOTATION):
        raise xpath_error('XPST0080')

    def nud_(self_: XPathFunction) -> XPathFunction:
        self_.parser.advance('(')
        self_[0:] = (self_.parser.expression(5),)
        self_.parser.advance(')')
        try:
            self_.value = self_.evaluate()
        except MissingContextError:
            self_.value = None
        return self_

    def evaluate_(self_: XPathFunction, context: Optional[XPathContext]=None) -> Union[List[None], AtomicValueType]:
        arg = self_.get_argument(context)
        if arg is None or self_.parser.schema is None:
            return []
        value = self_.string_value(arg)
        try:
            return self_.parser.schema.cast_as(value, atomic_type_name)
        except (TypeError, ValueError) as err:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self_.error('FORG0001', err)
    symbol = get_prefixed_name(atomic_type_name, self.namespaces)
    token_class_name = '_%sConstructorFunction' % symbol.replace(':', '_')
    kwargs = {'symbol': symbol, 'nargs': 1, 'label': 'constructor function', 'pattern': '\\b%s(?=\\s*\\(|\\s*\\(\\:.*\\:\\)\\()' % symbol, 'lbp': bp, 'rbp': bp, 'nud': nud_, 'evaluate': evaluate_, '__module__': self.__module__, '__qualname__': token_class_name, '__return__': None}
    token_class = cast(Type[XPathFunction], ABCMeta(token_class_name, (XPathFunction,), kwargs))
    MutableSequence.register(token_class)
    if self.symbol_table is self.__class__.symbol_table:
        self.symbol_table = dict(self.__class__.symbol_table)
    self.symbol_table[symbol] = token_class
    self.tokenizer = None
    return token_class