import pyglet
from pyglet.gl import *
def attach_texture_layer(self, texture, layer, level, target=GL_FRAMEBUFFER, attachment=GL_COLOR_ATTACHMENT0):
    """Attach a Texture layer to the Framebuffer

        :Parameters:
            `texture` : pyglet.image.TextureArray
                Specifies the texture object to attach to the framebuffer attachment
                point named by attachment.
            `layer` : int
                Specifies the layer of texture to attach.
            `level` : int
                Specifies the mipmap level of texture to attach.
            `target` : int
                Specifies the framebuffer target. target must be GL_DRAW_FRAMEBUFFER,
                GL_READ_FRAMEBUFFER, or GL_FRAMEBUFFER. GL_FRAMEBUFFER is equivalent
                to GL_DRAW_FRAMEBUFFER.
            `attachment` : int
                Specifies the attachment point of the framebuffer. attachment must be
                GL_COLOR_ATTACHMENTi, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT or
                GL_DEPTH_STENCIL_ATTACHMENT.
        """
    self.bind()
    glFramebufferTextureLayer(target, attachment, texture.id, level, layer)
    self._attachment_types |= attachment
    self._width = max(texture.width, self._width)
    self._height = max(texture.height, self._height)
    self.unbind()