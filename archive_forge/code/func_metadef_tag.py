from glance.api.v2 import image_members
from glance.api.v2 import images
from glance.api.v2 import metadef_namespaces
from glance.api.v2 import metadef_objects
from glance.api.v2 import metadef_properties
from glance.api.v2 import metadef_resource_types
from glance.api.v2 import metadef_tags
from glance.api.v2 import tasks
from glance.common import wsgi
def metadef_tag(self, req):
    return self.metadef_tag_schema.raw()