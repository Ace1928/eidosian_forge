import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def get_document_service(self):
    return DocumentServiceConnection(domain=self)