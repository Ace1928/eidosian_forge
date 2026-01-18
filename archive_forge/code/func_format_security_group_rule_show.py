from osc_lib import exceptions
from openstackclient.i18n import _
def format_security_group_rule_show(obj):
    data = transform_compute_security_group_rule(obj)
    return zip(*sorted(data.items()))