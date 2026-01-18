from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def gql_query(self):
    if self.gql_query_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.gql_query_ is None:
                self.gql_query_ = GqlQuery()
        finally:
            self.lazy_init_lock_.release()
    return self.gql_query_