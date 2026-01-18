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
def get_field_type_map(self) -> Dict[str, Tuple[Field, bool]]:
    if self._doc_field_type_map == {}:
        self._doc_field_type_map = {}
        for field in self.doc_field_types:
            for name in field.names:
                self._doc_field_type_map[name] = (field, False)
            if field.is_typed:
                typed_field = cast(TypedField, field)
                for name in typed_field.typenames:
                    self._doc_field_type_map[name] = (field, True)
    return self._doc_field_type_map