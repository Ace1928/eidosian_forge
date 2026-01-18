from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
def skipped_results_compiled_cursor(self):
    if self.skipped_results_compiled_cursor_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.skipped_results_compiled_cursor_ is None:
                self.skipped_results_compiled_cursor_ = CompiledCursor()
        finally:
            self.lazy_init_lock_.release()
    return self.skipped_results_compiled_cursor_