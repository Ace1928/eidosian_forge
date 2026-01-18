import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def register_container_instance(self, cluster=None, instance_identity_document=None, instance_identity_document_signature=None, total_resources=None):
    """
        This action is only used by the Amazon EC2 Container Service
        agent, and it is not intended for use outside of the agent.


        Registers an Amazon EC2 instance into the specified cluster.
        This instance will become available to place containers on.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that you want to register your container instance with.
            If you do not specify a cluster, the default cluster is assumed..

        :type instance_identity_document: string
        :param instance_identity_document:

        :type instance_identity_document_signature: string
        :param instance_identity_document_signature:

        :type total_resources: list
        :param total_resources:

        """
    params = {}
    if cluster is not None:
        params['cluster'] = cluster
    if instance_identity_document is not None:
        params['instanceIdentityDocument'] = instance_identity_document
    if instance_identity_document_signature is not None:
        params['instanceIdentityDocumentSignature'] = instance_identity_document_signature
    if total_resources is not None:
        self.build_complex_list_params(params, total_resources, 'totalResources.member', ('name', 'type', 'doubleValue', 'longValue', 'integerValue', 'stringSetValue'))
    return self._make_request(action='RegisterContainerInstance', verb='POST', path='/', params=params)