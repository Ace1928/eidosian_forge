from traitlets import Bool, validate
from .Material_autogen import Material as MaterialAutogen
@validate('needsUpdate')
def onNeedsUpdate(self, proposal):
    if proposal.value:
        content = {'type': 'needsUpdate'}
        self.send(content=content, buffers=None)
    return False