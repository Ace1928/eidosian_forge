from __future__ import unicode_literals
from commonmark.render.renderer import Renderer
def softbreak(self, node, entering):
    self.cr()