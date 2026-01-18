import base64
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def is_same_config(old, new):
    old_stripped = re.sub('\\s+', ' ', old, flags=re.S).rstrip()
    new_stripped = re.sub('\\s+', ' ', new, flags=re.S).rstrip()
    return old_stripped == new_stripped