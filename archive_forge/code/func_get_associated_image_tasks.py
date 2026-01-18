import testtools
def get_associated_image_tasks(self, *args, **kwargs):
    resource = self.controller.get_associated_image_tasks(*args, **kwargs)
    self._assertRequestId(resource)
    return resource