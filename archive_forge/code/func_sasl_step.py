import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def sasl_step(self, connection, pn_sasl):
    if 'PLAIN' in self.sasl_mechanisms:
        credentials = pn_sasl.recv()
        if not credentials:
            return
        if credentials not in self.user_credentials:
            return pn_sasl.done(pn_sasl.AUTH)
    pn_sasl.done(pn_sasl.OK)