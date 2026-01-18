from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
def parse_live_resource_data(self, resource_properties, resource_data):
    record_reality = {}
    rule = self.alarm_type + '_rule'
    threshold_data = resource_data.get(rule).copy()
    threshold_data.update(resource_data)
    for key in self.properties_schema.keys():
        if key in alarm_base.INTERNAL_PROPERTIES:
            continue
        if self.properties_schema[key].update_allowed:
            record_reality.update({key: threshold_data.get(key)})
    return record_reality