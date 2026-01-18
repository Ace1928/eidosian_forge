import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
def default_namespace_for_request():
    return None