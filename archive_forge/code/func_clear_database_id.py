from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_database_id(self):
    if self.has_database_id_:
        self.has_database_id_ = 0
        self.database_id_ = ''