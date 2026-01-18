import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def document_input(self, section, example, prefix, shape):
    input_section = section.add_new_section('input')
    input_section.style.start_codeblock()
    if prefix is not None:
        input_section.write(prefix)
    params = example.get('input', {})
    comments = example.get('comments')
    if comments:
        comments = comments.get('input')
    param_section = input_section.add_new_section('parameters')
    self._document_params(param_section, params, comments, [], shape)
    closing_section = input_section.add_new_section('input-close')
    closing_section.style.new_line()
    closing_section.style.new_line()
    closing_section.write('print(response)')
    closing_section.style.end_codeblock()