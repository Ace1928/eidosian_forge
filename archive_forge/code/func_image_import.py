import testtools
def image_import(self, *args, **kwargs):
    resp = self.controller.image_import(*args, **kwargs)
    self._assertRequestId(resp)