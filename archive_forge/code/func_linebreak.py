from __future__ import unicode_literals
from commonmark.render.renderer import Renderer
def linebreak(self, node, entering):
    self.cr()
    self.cr()