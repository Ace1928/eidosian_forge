from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.linode import get_user_agent
def maybe_instance_from_label(module, client):
    """Try to retrieve an instance based on a label."""
    try:
        label = module.params['label']
        result = client.linode.instances(Instance.label == label)
        return result[0]
    except IndexError:
        return None
    except Exception as exception:
        module.fail_json(msg='Unable to query the Linode API. Saw: %s' % exception)