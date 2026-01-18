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
def expected_next(self, *symbols: str, message: Optional[str]=None) -> None:
    """
        Checks the next token with a list of symbols. Replaces the next token with
        a '(name)' token if the check fails and the next token can be a name,
        otherwise raises a syntax error.

        :param symbols: a sequence of symbols.
        :param message: optional error message.
        """
    if self.next_token.symbol in symbols:
        return
    elif '(name)' in symbols and (not isinstance(self.next_token, (XPathFunction, XPathAxis))) and (self.name_pattern.match(self.next_token.symbol) is not None):
        self.next_token = self.symbol_table['(name)'](self, self.next_token.symbol)
    else:
        raise self.next_token.wrong_syntax(message)