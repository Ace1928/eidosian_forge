import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
def fake_gen():
    fake_data = {'cheddar': ['sharp', 'mild'], 'swiss': ['with holes', 'with lots of holes'], 'american': ['orange', 'yellow']}
    return mwdv.ModelWithDefaultViews(data=fake_data)