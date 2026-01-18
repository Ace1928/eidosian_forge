from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def export_as_image(self, *args, **kwargs):
    """Return an core :class:`~kivy.core.image.Image` of the actual
        widget.

        .. versionadded:: 1.11.0
        """
    from kivy.core.image import Image
    scale = kwargs.get('scale', 1)
    if self.parent is not None:
        canvas_parent_index = self.parent.canvas.indexof(self.canvas)
        if canvas_parent_index > -1:
            self.parent.canvas.remove(self.canvas)
    fbo = Fbo(size=(self.width * scale, self.height * scale), with_stencilbuffer=True)
    with fbo:
        ClearColor(0, 0, 0, 0)
        ClearBuffers()
        Scale(1, -1, 1)
        Scale(scale, scale, 1)
        Translate(-self.x, -self.y - self.height, 0)
    fbo.add(self.canvas)
    fbo.draw()
    img = Image(fbo.texture)
    fbo.remove(self.canvas)
    if self.parent is not None and canvas_parent_index > -1:
        self.parent.canvas.insert(canvas_parent_index, self.canvas)
    return img