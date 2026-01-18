from __future__ import annotations
import ast
import datetime
import os
import re
import sys
from io import BytesIO, TextIOWrapper
import yaml
import yaml.reader
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.yaml import SafeLoader
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.loader import AnsibleLoader
def parse_yaml(value, lineno, module, name, load_all=False, ansible_loader=False):
    traces = []
    errors = []
    data = None
    if load_all:
        yaml_load = yaml.load_all
    else:
        yaml_load = yaml.load
    if ansible_loader:
        loader = AnsibleLoader
    else:
        loader = SafeLoader
    try:
        data = yaml_load(value, Loader=loader)
        if load_all:
            data = list(data)
    except yaml.MarkedYAMLError as e:
        errors.append({'msg': '%s is not valid YAML' % name, 'line': e.problem_mark.line + lineno, 'column': e.problem_mark.column + 1})
        traces.append(e)
    except yaml.reader.ReaderError as e:
        traces.append(e)
        errors.append({'msg': '%s is not valid YAML. Character 0x%x at position %d.' % (name, e.character, e.position), 'line': lineno})
    except yaml.YAMLError as e:
        traces.append(e)
        errors.append({'msg': '%s is not valid YAML: %s: %s' % (name, type(e), e), 'line': lineno})
    return (data, errors, traces)