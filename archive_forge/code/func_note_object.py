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
def note_object(self, objtype: str, name: str, node_id: str, location: Any=None) -> None:
    if (objtype, name) in self.objects:
        docname, node_id = self.objects[objtype, name]
        logger.warning(__('duplicate description of %s %s, other instance in %s') % (objtype, name, docname), location=location)
    self.objects[objtype, name] = (self.env.docname, node_id)