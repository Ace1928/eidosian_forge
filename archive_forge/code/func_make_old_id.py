import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils.nodes import Element
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import OptionSpec
def make_old_id(self, name: str) -> str:
    """Generate old styled node_id for directive options.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
    return '-'.join([self.objtype, self.current_directive, name])