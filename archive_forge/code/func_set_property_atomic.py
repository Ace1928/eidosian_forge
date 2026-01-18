def set_property_atomic(self, item, name, value):
    msg = '%s is only valid for images' % __name__
    assert hasattr(item, 'image_id'), msg
    self.base.set_property_atomic(item, name, value)