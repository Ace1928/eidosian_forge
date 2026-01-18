import inspect
import types
from botocore.docs.example import (
from botocore.docs.params import (
def document_custom_method(section, method_name, method):
    """Documents a non-data driven method

    :param section: The section to write the documentation to.

    :param method_name: The name of the method

    :param method: The handle to the method being documented
    """
    full_method_name = f'{section.context.get('qualifier', '')}{method_name}'
    document_custom_signature(section, full_method_name, method)
    method_intro_section = section.add_new_section('method-intro')
    method_intro_section.writeln('')
    doc_string = inspect.getdoc(method)
    if doc_string is not None:
        method_intro_section.style.write_py_doc_string(doc_string)