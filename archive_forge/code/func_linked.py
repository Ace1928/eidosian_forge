from unittest import mock
import eventlet
from oslo_context import context
from heat.engine import service
from heat.tests import common
def linked(gt, thread):
    for i in range(10):
        eventlet.sleep()
    done.append(thread)