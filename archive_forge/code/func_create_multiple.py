import testtools
def create_multiple(self, *args, **kwargs):
    tags = self.controller.create_multiple(*args, **kwargs)
    actual = [tag.name for tag in tags]
    self._assertRequestId(tags)
    return actual