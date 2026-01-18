from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_container_dict_from_obj(container):
    """
    Create a dict from an instance of a Container.

    :param rule: Container
    :return: dict
    """
    results = dict(name=container.name, image=container.image, memory=container.resources.requests.memory_in_gb, cpu=container.resources.requests.cpu)
    if container.instance_view is not None:
        results['instance_restart_count'] = container.instance_view.restart_count
        if container.instance_view.current_state:
            results['instance_current_state'] = container.instance_view.current_state.state
            results['instance_current_start_time'] = container.instance_view.current_state.start_time
            results['instance_current_exit_code'] = container.instance_view.current_state.exit_code
            results['instance_current_finish_time'] = container.instance_view.current_state.finish_time
            results['instance_current_detail_status'] = container.instance_view.current_state.detail_status
        if container.instance_view.previous_state:
            results['instance_previous_state'] = container.instance_view.previous_state.state
            results['instance_previous_start_time'] = container.instance_view.previous_state.start_time
            results['instance_previous_exit_code'] = container.instance_view.previous_state.exit_code
            results['instance_previous_finish_time'] = container.instance_view.previous_state.finish_time
            results['instance_previous_detail_status'] = container.instance_view.previous_state.detail_status
    return results