import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def document_shared_example(self, example, prefix, section, operation_model):
    """Documents a single shared example based on its definition.

        :param example: The model of the example

        :param prefix: The prefix to use in the method example.

        :param section: The section to write to.

        :param operation_model: The model of the operation used in the example
        """
    section.style.new_paragraph()
    section.write(example.get('description'))
    section.style.new_line()
    self.document_input(section, example, prefix, operation_model.input_shape)
    self.document_output(section, example, operation_model.output_shape)