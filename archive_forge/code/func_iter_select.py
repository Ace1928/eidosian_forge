from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Union, Type
from .namespaces import NamespacesType
from .tree_builders import RootArgType
from .xpath_context import XPathContext
from .xpath2 import XPath2Parser
def iter_select(self, root: Optional[RootArgType], **kwargs: Any) -> Iterator[Any]:
    """
        Creates an XPath selector generator for apply the instance's XPath expression
        on *root* Element.

        :param root: the root of the XML document, usually an ElementTree instance         or an Element.
        :param kwargs: other optional parameters for the XPath dynamic context.
        :return: a generator of the XPath expression results.
        """
    if 'variables' not in kwargs and self._variables:
        kwargs['variables'] = self._variables
    context = XPathContext(root, **kwargs)
    return self.root_token.select_results(context)