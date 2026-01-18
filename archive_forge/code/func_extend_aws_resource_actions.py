from ansible.module_utils._text import to_native
from ansible.plugins.callback import CallbackBase
def extend_aws_resource_actions(self, result):
    if result.get('resource_actions'):
        self.aws_resource_actions.extend(result['resource_actions'])