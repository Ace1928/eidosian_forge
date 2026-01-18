from botocore import xform_name
from botocore.docs.method import get_instance_public_methods
from botocore.docs.utils import DocumentedShape
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import add_resource_type_overview
def document_collection_object(section, collection_model, include_signature=True):
    """Documents a collection resource object

    :param section: The section to write to

    :param collection_model: The model of the collection

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    if include_signature:
        section.style.start_sphinx_py_attr(collection_model.name)
    section.include_doc_string('A collection of %s resources.' % collection_model.resource.type)
    section.include_doc_string('A %s Collection will include all resources by default, and extreme caution should be taken when performing actions on all resources.' % collection_model.resource.type)