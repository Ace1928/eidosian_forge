import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def document_shared_examples(section, operation_model, example_prefix, shared_examples):
    """Documents the shared examples

    :param section: The section to write to.

    :param operation_model: The model of the operation.

    :param example_prefix: The prefix to use in the method example.

    :param shared_examples: The shared JSON examples from the model.
    """
    container_section = section.add_new_section('shared-examples')
    container_section.style.new_paragraph()
    container_section.style.bold('Examples')
    documenter = SharedExampleDocumenter()
    for example in shared_examples:
        documenter.document_shared_example(example=example, section=container_section.add_new_section(example['id']), prefix=example_prefix, operation_model=operation_model)