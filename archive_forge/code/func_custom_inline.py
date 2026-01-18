from __future__ import unicode_literals
import re
from builtins import str
from commonmark.common import escape_xml
from commonmark.render.renderer import Renderer
def custom_inline(self, node, entering):
    if entering and node.on_enter:
        self.lit(node.on_enter)
    elif not entering and node.on_exit:
        self.lit(node.on_exit)