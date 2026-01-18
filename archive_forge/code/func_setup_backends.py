from oslo_log import log
from keystone.common import sql
import keystone.conf
from keystone.server import backends
def setup_backends(load_extra_backends_fn=lambda: {}, startup_application_fn=lambda: None):
    drivers = backends.load_backends()
    drivers.update(load_extra_backends_fn())
    res = startup_application_fn()
    return (drivers, res)