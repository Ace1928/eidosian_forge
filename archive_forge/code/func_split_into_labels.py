from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.common.text.converters import to_text
def split_into_labels(domain):
    """
    Split domain name to a list of labels. Start with the top-most label.

    Returns a list of labels and a tail, which is either ``''`` or ``'.'``.
    Raises ``InvalidDomainName`` if the domain name is not valid.
    """
    result = []
    index = len(domain)
    tail = ''
    if domain.endswith('.'):
        index -= 1
        tail = '.'
    if index > 0:
        while index >= 0:
            next_index = domain.rfind('.', 0, index)
            label = domain[next_index + 1:index]
            if label == '' or label[0] == '-' or label[-1] == '-' or (len(label) > 63):
                raise InvalidDomainName(domain)
            result.append(label)
            index = next_index
    return (result, tail)