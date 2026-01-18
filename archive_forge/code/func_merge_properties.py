import jsonschema
from oslo_utils import encodeutils
from glance.common import exception
from glance.i18n import _
def merge_properties(self, properties):
    original_keys = set(self.properties.keys())
    new_keys = set(properties.keys())
    intersecting_keys = original_keys.intersection(new_keys)
    conflicting_keys = [k for k in intersecting_keys if self.properties[k] != properties[k]]
    if conflicting_keys:
        props = ', '.join(conflicting_keys)
        reason = _('custom properties (%(props)s) conflict with base properties')
        raise exception.SchemaLoadError(reason=reason % {'props': props})
    self.properties.update(properties)