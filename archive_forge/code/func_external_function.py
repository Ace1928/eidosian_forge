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
def external_function(self, callback: Callable[..., Any], name: Optional[str]=None, prefix: Optional[str]=None, sequence_types: Tuple[str, ...]=(), bp: int=90) -> Type[XPathFunction]:
    """Registers a token class for an external function."""
    import inspect
    symbol = name or callback.__name__
    if not is_ncname(symbol):
        raise ElementPathValueError(f'{symbol!r} is not a name')
    elif symbol in self.RESERVED_FUNCTION_NAMES:
        raise ElementPathValueError(f'{symbol!r} is a reserved function name')
    nargs: NargsType
    spec = inspect.getfullargspec(callback)
    if spec.varargs is not None:
        if spec.args:
            nargs = (len(spec.args), None)
        else:
            nargs = None
    elif spec.defaults is None:
        nargs = len(spec.args)
    else:
        nargs = (len(spec.args) - len(spec.defaults), len(spec.args))
    if prefix:
        namespace = self.namespaces[prefix]
        qname = QName(namespace, f'{prefix}:{symbol}')
    else:
        namespace = XPATH_FUNCTIONS_NAMESPACE
        qname = QName(XPATH_FUNCTIONS_NAMESPACE, f'fn:{symbol}')
    class_name = f'{upper_camel_case(qname.qname)}ExternalFunction'
    lookup_name = qname.expanded_name
    if self.symbol_table is self.__class__.symbol_table:
        self.symbol_table = dict(self.__class__.symbol_table)
    if lookup_name in self.symbol_table:
        msg = f'function {qname.qname!r} is already registered'
        raise ElementPathValueError(msg)
    elif symbol not in self.symbol_table or not issubclass(self.symbol_table[symbol], ProxyToken):
        if symbol in self.symbol_table:
            token_cls = self.symbol_table[symbol]
            if not issubclass(token_cls, XPathFunction) or token_cls.label == 'kind test':
                msg = f'{symbol!r} name collides with {token_cls!r}'
                raise ElementPathValueError(msg)
            if namespace == token_cls.namespace:
                msg = f'function {qname.qname!r} is already registered'
                raise ElementPathValueError(msg)
            self.symbol_table[f'{{{token_cls.namespace}}}{symbol}'] = token_cls
        proxy_class_name = f'{upper_camel_case(qname.local_name)}ProxyToken'
        kwargs = {'class_name': proxy_class_name, 'symbol': symbol, 'lbp': bp, 'rbp': bp, '__module__': self.__module__, '__qualname__': proxy_class_name, '__return__': None}
        proxy_class = cast(Type[ProxyToken], ABCMeta(class_name, (ProxyToken,), kwargs))
        MutableSequence.register(proxy_class)
        self.symbol_table[symbol] = proxy_class

    def evaluate_external_function(self_: XPathFunction, context: Optional[XPathContext]=None) -> Any:
        args = []
        for k in range(len(self_)):
            arg = self_.get_argument(context, index=k)
            args.append(arg)
        if sequence_types:
            for k, (arg, st) in enumerate(zip(args, sequence_types), start=1):
                if not match_sequence_type(arg, st, self):
                    msg_ = f'{ordinal(k)} argument does not match sequence type {st!r}'
                    raise xpath_error('XPDY0050', msg_)
            result = callback(*args)
            if not match_sequence_type(result, sequence_types[-1], self):
                msg_ = f'Result does not match sequence type {sequence_types[-1]!r}'
                raise xpath_error('XPDY0050', msg_)
            return result
        return callback(*args)
    kwargs = {'class_name': class_name, 'symbol': symbol, 'namespace': namespace, 'label': 'external function', 'nargs': nargs, 'lbp': bp, 'rbp': bp, 'evaluate': evaluate_external_function, '__module__': self.__module__, '__qualname__': class_name, '__return__': None}
    if sequence_types:
        kwargs['sequence_types'] = sequence_types
        if self.function_signatures is self.__class__.function_signatures:
            self.function_signatures = dict(self.__class__.function_signatures)
        if nargs is None:
            pass
        elif isinstance(nargs, int):
            assert len(sequence_types) == nargs + 1
            self.function_signatures[qname, nargs] = 'function({}) as {}'.format(', '.join(sequence_types[:-1]), sequence_types[-1])
        elif nargs[1] is None:
            assert len(sequence_types) == nargs[0] + 1
            self.function_signatures[qname, nargs[0]] = 'function({}, ...) as {}'.format(', '.join(sequence_types[:-1]), sequence_types[-1])
        else:
            assert len(sequence_types) == nargs[1] + 1
            for arity in range(nargs[0], nargs[1] + 1):
                self.function_signatures[qname, arity] = 'function({}) as {}'.format(', '.join(sequence_types[:arity]), sequence_types[-1])
    token_class = cast(Type[XPathFunction], ABCMeta(class_name, (XPathFunction,), kwargs))
    MutableSequence.register(token_class)
    self.symbol_table[lookup_name] = token_class
    self.tokenizer = None
    return token_class