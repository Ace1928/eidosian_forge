import os
import routes
import webob
from glance.api.middleware import context
from glance.api.v2 import router
import glance.common.client
def fake_reader():
    return res.body