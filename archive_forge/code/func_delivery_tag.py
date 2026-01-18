from .serialization import GenericContent
from .spec import Basic
@property
def delivery_tag(self):
    return self.delivery_info.get('delivery_tag')