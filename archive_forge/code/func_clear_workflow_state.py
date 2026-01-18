from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_workflow_state(self):
    if self.has_workflow_state_:
        self.has_workflow_state_ = 0
        self.workflow_state_ = 0