from copy import copy
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XSD_NAMESPACE
from ..xpath_nodes import AttributeNode, ElementNode
from ..xpath_tokens import XPathToken, ValueToken, XPathFunction, \
from ..xpath_context import XPathSchemaContext
from ..datatypes import QName
from .xpath30_parser import XPath30Parser
@method('let')
def select_let_expression(self, context=None):
    if context is None:
        raise self.missing_context()
    for k in range(0, len(self) - 1, 2):
        varname = self[k][0].value
        value = self[k + 1].evaluate(context)
        context.variables[varname] = value
    yield from self[-1].select(context)