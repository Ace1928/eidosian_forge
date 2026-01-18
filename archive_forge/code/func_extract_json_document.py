from __future__ import absolute_import, division, print_function
import shlex
import pipes
import re
import json
import os
def extract_json_document(output):
    """
    This is for specific type of mongo shell return data in the format SomeText()
    https://github.com/ansible-collections/community.mongodb/issues/436
    i.e.

    """
    output = output.strip()
    if re.match('^[a-zA-Z].*\\(', output) and output.endswith(')'):
        first_bracket = output.find('{')
        last_bracket = output.rfind('}')
        if first_bracket > 0 and last_bracket > 0:
            tmp = output[first_bracket:last_bracket + 1]
            tmp = tmp.replace('\n', '')
            tmp = tmp.replace('\t', '')
            if tmp is not None:
                output = tmp
    return output