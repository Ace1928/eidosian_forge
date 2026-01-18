import re
from abc import ABCMeta
from typing import cast, Any, ClassVar, Dict, MutableMapping, \
from ..exceptions import MissingContextError, ElementPathValueError, \
from ..datatypes import QName
from ..tdop import Token, Parser
from ..namespaces import NamespacesType, XML_NAMESPACE, XSD_NAMESPACE, \
from ..sequence_types import match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_tokens import NargsType, XPathToken, XPathAxis, XPathFunction, \
@classmethod
def create_restricted_parser(cls, name: str, symbols: Sequence[str]) -> Type['XPath1Parser']:
    """Get a parser subclass with a restricted set of symbols.s"""
    symbol_table = {k: v for k, v in cls.symbol_table.items() if k in symbols}
    return cast(Type['XPath1Parser'], ABCMeta(f'{name}{cls.__name__}', (cls,), {'symbol_table': symbol_table}))