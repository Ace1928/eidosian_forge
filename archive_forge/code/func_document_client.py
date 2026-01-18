import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def document_client(self, section):
    """Documents a client and its methods

        :param section: The section to write to.
        """
    self._add_title(section)
    self._add_class_signature(section)
    client_methods = self._get_client_methods()
    self._add_client_intro(section, client_methods)
    self._add_client_methods(client_methods)