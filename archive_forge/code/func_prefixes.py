from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import tempfile
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
@property
def prefixes(self):
    return ['google.datastore.v1.Datastore', 'google.datastore.v1.AllocateIdsRequest', 'google.datastore.v1.AllocateIdsResponse', 'google.datastore.v1.ArrayValue', 'google.datastore.v1.BeginTransactionRequest', 'google.datastore.v1.BeginTransactionResponse', 'google.datastore.v1.CommitRequest', 'google.datastore.v1.CommitRequest.Mode', 'google.datastore.v1.CommitResponse', 'google.datastore.v1.CompositeFilter', 'google.datastore.v1.CompositeFilter.Operator', 'google.datastore.v1.Entity', 'google.datastore.v1.EntityResult', 'google.datastore.v1.EntityResult.ResultType', 'google.datastore.v1.Filter', 'google.datastore.v1.GqlQuery', 'google.datastore.v1.GqlQueryParameter', 'google.datastore.v1.Key', 'google.datastore.v1.Key.PathElement', 'google.datastore.v1.KindExpression', 'google.datastore.v1.LookupRequest', 'google.datastore.v1.LookupResponse', 'google.datastore.v1.Mutation', 'google.datastore.v1.MutationResult', 'google.datastore.v1.PartitionId', 'google.datastore.v1.Projection', 'google.datastore.v1.PropertyFilter', 'google.datastore.v1.PropertyFilter.Operator', 'google.datastore.v1.PropertyOrder', 'google.datastore.v1.PropertyOrder.Direction', 'google.datastore.v1.PropertyReference', 'google.datastore.v1.Query', 'google.datastore.v1.QueryResultBatch', 'google.datastore.v1.QueryResultBatch.MoreResultsType', 'google.datastore.v1.ReadOptions', 'google.datastore.v1.ReadOptions.ReadConsistencygoogle.datastore.v1.RollbackRequest', 'google.datastore.v1.RollbackResponse', 'google.datastore.v1.RunQueryRequest', 'google.datastore.v1.RunQueryResponse', 'google.datastore.v1.Value']