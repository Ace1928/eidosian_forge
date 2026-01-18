from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def detect_task_results(results):
    if 'results' in results:
        for key in ('changed', 'msg', 'skipped'):
            if key not in results:
                raise ValueError(f'missing {key} key')
        if not isinstance(results['results'], list):
            raise ValueError('results is present, but not a list')
        for index, result in enumerate(results['results']):
            if not isinstance(result, dict):
                raise ValueError(f'result {index + 1} is not a dictionary')
            for key in ('changed', 'failed', 'ansible_loop_var', 'invocation'):
                if key not in result:
                    raise ValueError(f'missing {key} key for result {index + 1}')
            yield (f' for result #{index + 1}', result)
        return
    for key in ('changed', 'failed'):
        if key not in results:
            raise ValueError(f'missing {key} key')
    yield ('', results)