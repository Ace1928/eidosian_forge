import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
def describe_signature_as_introducer(self, parentNode: desc_signature, mode: str, env: 'BuildEnvironment', symbol: 'Symbol', lineSpec: bool) -> None:
    signode = addnodes.desc_signature_line()
    parentNode += signode
    signode.sphinx_line_type = 'templateIntroduction'
    self.concept.describe_signature(signode, 'markType', env, symbol)
    signode += addnodes.desc_sig_punctuation('{', '{')
    first = True
    for param in self.params:
        if not first:
            signode += addnodes.desc_sig_punctuation(',', ',')
            signode += addnodes.desc_sig_space()
        first = False
        param.describe_signature(signode, mode, env, symbol)
    signode += addnodes.desc_sig_punctuation('}', '}')