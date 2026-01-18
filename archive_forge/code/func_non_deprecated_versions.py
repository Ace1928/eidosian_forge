from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
@property
def non_deprecated_versions(self):
    """Retrieves the versions of the library that are not deprecated.

    Returns:
      A list of the versions of the library that are not deprecated.
    """
    return [version for version in self.supported_versions if version not in self.deprecated_versions]