import argparse
import functools
import hashlib
import logging
import os
from oslo_utils import encodeutils
from oslo_utils import importutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
def convert_to_lowercase(string):
    return string.lower()