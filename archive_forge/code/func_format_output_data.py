import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def format_output_data(self, data):
    rules = []
    for rule in data['policy'].get('rules', []):
        rules.append('%s (type: %s)' % (rule['id'], rule['type']))
    data['policy']['rules'] = os.linesep.join(rules)
    super(ShowQoSPolicy, self).format_output_data(data)