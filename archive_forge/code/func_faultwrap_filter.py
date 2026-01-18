from heat.api.middleware import fault
from heat.api.middleware import version_negotiation as vn
from heat.api.openstack import versions
def faultwrap_filter(app, conf, **local_conf):
    return fault.FaultWrapper(app)