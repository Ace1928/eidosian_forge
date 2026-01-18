from keystoneauth1 import _utils as utils
def add_media_type(self, base, type):
    mt = {'base': base, 'type': type}
    self.media_types.append(mt)
    return mt