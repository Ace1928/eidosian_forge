import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def fix_ids(self, tree: nodes.document) -> None:
    """Replace colons with hyphens in href and id attributes.

        Some readers crash because they interpret the part as a
        transport protocol specification.
        """

    def update_node_id(node: Element) -> None:
        """Update IDs of given *node*."""
        new_ids: List[str] = []
        for node_id in node['ids']:
            new_id = self.fix_fragment('', node_id)
            if new_id not in new_ids:
                new_ids.append(new_id)
        node['ids'] = new_ids
    for reference in tree.findall(nodes.reference):
        if 'refuri' in reference:
            m = self.refuri_re.match(reference['refuri'])
            if m:
                reference['refuri'] = self.fix_fragment(m.group(1), m.group(2))
        if 'refid' in reference:
            reference['refid'] = self.fix_fragment('', reference['refid'])
    for target in tree.findall(nodes.target):
        update_node_id(target)
        next_node: Node = target.next_node(ascend=True)
        if isinstance(next_node, nodes.Element):
            update_node_id(next_node)
    for desc_signature in tree.findall(addnodes.desc_signature):
        update_node_id(desc_signature)