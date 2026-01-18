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
def remove_route(self, address, link):
    if address in self._sources:
        if link in self._sources[address]:
            self._sources[address].remove(link)
            if not self._sources[address]:
                del self._sources[address]