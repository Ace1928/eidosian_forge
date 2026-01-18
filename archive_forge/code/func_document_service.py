import os
import boto3
from botocore.exceptions import DataNotFoundError
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.docs.bcdoc.restdoc import DocumentStructure
from boto3.utils import ServiceContext
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter
from boto3.docs.resource import ServiceResourceDocumenter
def document_service(self):
    """Documents an entire service.

        :returns: The reStructured text of the documented service.
        """
    doc_structure = DocumentStructure(self._service_name, section_names=self.sections, target='html')
    self.title(doc_structure.get_section('title'))
    self.table_of_contents(doc_structure.get_section('table-of-contents'))
    self.client_api(doc_structure.get_section('client'))
    self.paginator_api(doc_structure.get_section('paginators'))
    self.waiter_api(doc_structure.get_section('waiters'))
    if self._service_resource:
        self._document_service_resource(doc_structure.get_section('service-resource'))
        self._document_resources(doc_structure.get_section('resources'))
    self._document_examples(doc_structure.get_section('examples'))
    return doc_structure.flush_structure()