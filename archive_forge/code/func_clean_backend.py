import contextlib
from kazoo import exceptions as kazoo_exceptions
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_zookeeper
from taskflow import test
from taskflow.tests.unit.persistence import base
from taskflow.tests import utils as test_utils
from taskflow.utils import kazoo_utils
def clean_backend(backend, conf):
    with contextlib.closing(backend.get_connection()) as conn:
        try:
            conn.clear_all()
        except exc.NotFound:
            pass
    client = kazoo_utils.make_client(conf)
    client.start()
    try:
        client.delete(conf['path'], recursive=True)
    except kazoo_exceptions.NoNodeError:
        pass
    finally:
        kazoo_utils.finalize_client(client)