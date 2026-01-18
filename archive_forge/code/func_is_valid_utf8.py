from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def is_valid_utf8(s):
    if isinstance(s, six.text_type):
        return True
    try:
        s.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False