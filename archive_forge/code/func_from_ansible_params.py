from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
@classmethod
def from_ansible_params(cls, ap, old_service, image_digest, secret_ids, config_ids, network_ids, docker_api_version, docker_py_version):
    s = DockerService(docker_api_version, docker_py_version)
    s.image = image_digest
    s.args = ap['args']
    s.endpoint_mode = ap['endpoint_mode']
    s.dns = ap['dns']
    s.dns_search = ap['dns_search']
    s.dns_options = ap['dns_options']
    s.healthcheck, s.healthcheck_disabled = parse_healthcheck(ap['healthcheck'])
    s.hostname = ap['hostname']
    s.hosts = ap['hosts']
    s.tty = ap['tty']
    s.labels = ap['labels']
    s.container_labels = ap['container_labels']
    s.mode = ap['mode']
    s.stop_signal = ap['stop_signal']
    s.user = ap['user']
    s.working_dir = ap['working_dir']
    s.read_only = ap['read_only']
    s.init = ap['init']
    s.cap_add = ap['cap_add']
    s.cap_drop = ap['cap_drop']
    s.networks = get_docker_networks(ap['networks'], network_ids)
    s.command = ap['command']
    if isinstance(s.command, string_types):
        s.command = shlex.split(s.command)
    elif isinstance(s.command, list):
        invalid_items = [(index, item) for index, item in enumerate(s.command) if not isinstance(item, string_types)]
        if invalid_items:
            errors = ', '.join(['%s (%s) at index %s' % (item, type(item), index) for index, item in invalid_items])
            raise Exception('All items in a command list need to be strings. Check quoting. Invalid items: %s.' % errors)
        s.command = ap['command']
    elif s.command is not None:
        raise ValueError('Invalid type for command %s (%s). Only string or list allowed. Check quoting.' % (s.command, type(s.command)))
    s.env = get_docker_environment(ap['env'], ap['env_files'])
    s.rollback_config = cls.get_rollback_config_from_ansible_params(ap)
    update_config = cls.get_update_config_from_ansible_params(ap)
    for key, value in update_config.items():
        setattr(s, key, value)
    restart_config = cls.get_restart_config_from_ansible_params(ap)
    for key, value in restart_config.items():
        setattr(s, key, value)
    logging_config = cls.get_logging_from_ansible_params(ap)
    for key, value in logging_config.items():
        setattr(s, key, value)
    limits = cls.get_limits_from_ansible_params(ap)
    for key, value in limits.items():
        setattr(s, key, value)
    reservations = cls.get_reservations_from_ansible_params(ap)
    for key, value in reservations.items():
        setattr(s, key, value)
    placement = cls.get_placement_from_ansible_params(ap)
    for key, value in placement.items():
        setattr(s, key, value)
    if ap['stop_grace_period'] is not None:
        s.stop_grace_period = convert_duration_to_nanosecond(ap['stop_grace_period'])
    if ap['force_update']:
        s.force_update = int(str(time.time()).replace('.', ''))
    if ap['groups'] is not None:
        s.groups = [str(g) for g in ap['groups']]
    if ap['replicas'] == -1:
        if old_service:
            s.replicas = old_service.replicas
        else:
            s.replicas = 1
    else:
        s.replicas = ap['replicas']
    if ap['publish'] is not None:
        s.publish = []
        for param_p in ap['publish']:
            service_p = {}
            service_p['protocol'] = param_p['protocol']
            service_p['mode'] = param_p['mode']
            service_p['published_port'] = param_p['published_port']
            service_p['target_port'] = param_p['target_port']
            s.publish.append(service_p)
    if ap['mounts'] is not None:
        s.mounts = []
        for param_m in ap['mounts']:
            service_m = {}
            service_m['readonly'] = param_m['readonly']
            service_m['type'] = param_m['type']
            if param_m['source'] is None and param_m['type'] != 'tmpfs':
                raise ValueError('Source must be specified for mounts which are not of type tmpfs')
            service_m['source'] = param_m['source'] or ''
            service_m['target'] = param_m['target']
            service_m['labels'] = param_m['labels']
            service_m['no_copy'] = param_m['no_copy']
            service_m['propagation'] = param_m['propagation']
            service_m['driver_config'] = param_m['driver_config']
            service_m['tmpfs_mode'] = param_m['tmpfs_mode']
            tmpfs_size = param_m['tmpfs_size']
            if tmpfs_size is not None:
                try:
                    tmpfs_size = human_to_bytes(tmpfs_size)
                except ValueError as exc:
                    raise ValueError('Failed to convert tmpfs_size to bytes: %s' % exc)
            service_m['tmpfs_size'] = tmpfs_size
            s.mounts.append(service_m)
    if ap['configs'] is not None:
        s.configs = []
        for param_m in ap['configs']:
            service_c = {}
            config_name = param_m['config_name']
            service_c['config_id'] = param_m['config_id'] or config_ids[config_name]
            service_c['config_name'] = config_name
            service_c['filename'] = param_m['filename'] or config_name
            service_c['uid'] = param_m['uid']
            service_c['gid'] = param_m['gid']
            service_c['mode'] = param_m['mode']
            s.configs.append(service_c)
    if ap['secrets'] is not None:
        s.secrets = []
        for param_m in ap['secrets']:
            service_s = {}
            secret_name = param_m['secret_name']
            service_s['secret_id'] = param_m['secret_id'] or secret_ids[secret_name]
            service_s['secret_name'] = secret_name
            service_s['filename'] = param_m['filename'] or secret_name
            service_s['uid'] = param_m['uid']
            service_s['gid'] = param_m['gid']
            service_s['mode'] = param_m['mode']
            s.secrets.append(service_s)
    return s