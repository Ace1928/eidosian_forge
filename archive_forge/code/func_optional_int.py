import re
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import addnodes
from sphinx.addnodes import desc_signature
from sphinx.util import docutils
from sphinx.util.docfields import DocFieldTransformer, Field, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
def optional_int(argument: str) -> int:
    """
    Check for an integer argument or None value; raise ``ValueError`` if not.
    """
    if argument is None:
        return None
    else:
        value = int(argument)
        if value < 0:
            raise ValueError('negative value; must be positive or zero')
        return value