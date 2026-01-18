import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
@fixture(autouse=True)
def term_context_instance(request):
    request.addfinalizer(lambda: term_context(zmq.Context.instance(), timeout=10))