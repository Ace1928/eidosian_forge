from __future__ import absolute_import, division, print_function
import json
import os
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (FINAL_STATUSES, rax_argument_spec, rax_find_bootable_volume,
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.six import string_types
def cloudservers(module, state=None, name=None, flavor=None, image=None, meta=None, key_name=None, files=None, wait=True, wait_timeout=300, disk_config=None, count=1, group=None, instance_ids=None, exact_count=False, networks=None, count_offset=0, auto_increment=False, extra_create_args=None, user_data=None, config_drive=False, boot_from_volume=False, boot_volume=None, boot_volume_size=None, boot_volume_terminate=False):
    meta = {} if meta is None else meta
    files = {} if files is None else files
    instance_ids = [] if instance_ids is None else instance_ids
    networks = [] if networks is None else networks
    extra_create_args = {} if extra_create_args is None else extra_create_args
    cs = pyrax.cloudservers
    cnw = pyrax.cloud_networks
    if not cnw:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if state == 'present' or (state == 'absent' and instance_ids is None):
        if not boot_from_volume and (not boot_volume) and (not image):
            module.fail_json(msg='image is required for the "rax" module')
        for arg, value in dict(name=name, flavor=flavor).items():
            if not value:
                module.fail_json(msg='%s is required for the "rax" module' % arg)
        if boot_from_volume and (not image) and (not boot_volume):
            module.fail_json(msg='image or boot_volume are required for the "rax" with boot_from_volume')
        if boot_from_volume and image and (not boot_volume_size):
            module.fail_json(msg='boot_volume_size is required for the "rax" module with boot_from_volume and image')
        if boot_from_volume and image and boot_volume:
            image = None
    servers = []
    if group and 'group' not in meta:
        meta['group'] = group
    elif 'group' in meta and group is None:
        group = meta['group']
    for k, v in meta.items():
        if isinstance(v, list):
            meta[k] = ','.join(['%s' % i for i in v])
        elif isinstance(v, dict):
            meta[k] = json.dumps(v)
        elif not isinstance(v, string_types):
            meta[k] = '%s' % v
    was_absent = False
    if group is not None and state == 'absent':
        exact_count = True
        state = 'present'
        was_absent = True
    if image:
        image = rax_find_image(module, pyrax, image)
    nics = []
    if networks:
        for network in networks:
            nics.extend(rax_find_network(module, pyrax, network))
    if state == 'present':
        if exact_count is not False:
            if group is None:
                module.fail_json(msg='"group" must be provided when using "exact_count"')
            if auto_increment:
                numbers = set()
                try:
                    name % 0
                except TypeError as e:
                    if e.message.startswith('not all'):
                        name = '%s%%d' % name
                    else:
                        module.fail_json(msg=e.message)
                pattern = re.sub('%\\d*[sd]', '(\\d+)', name)
                for server in cs.servers.list():
                    if server.status == 'DELETED':
                        continue
                    if server.metadata.get('group') == group:
                        servers.append(server)
                    match = re.search(pattern, server.name)
                    if match:
                        number = int(match.group(1))
                        numbers.add(number)
                number_range = xrange(count_offset, count_offset + count)
                available_numbers = list(set(number_range).difference(numbers))
            else:
                for server in cs.servers.list():
                    if server.status == 'DELETED':
                        continue
                    if server.metadata.get('group') == group:
                        servers.append(server)
            if was_absent:
                diff = len(servers) - count
                if diff < 0:
                    count = 0
                else:
                    count = diff
            if len(servers) > count:
                state = 'absent'
                kept = servers[:count]
                del servers[:count]
                instance_ids = []
                for server in servers:
                    instance_ids.append(server.id)
                delete(module, instance_ids=instance_ids, wait=wait, wait_timeout=wait_timeout, kept=kept)
            elif len(servers) < count:
                if auto_increment:
                    names = []
                    name_slice = count - len(servers)
                    numbers_to_use = available_numbers[:name_slice]
                    for number in numbers_to_use:
                        names.append(name % number)
                else:
                    names = [name] * (count - len(servers))
            else:
                instances = []
                instance_ids = []
                for server in servers:
                    instances.append(rax_to_dict(server, 'server'))
                    instance_ids.append(server.id)
                module.exit_json(changed=False, action=None, instances=instances, success=[], error=[], timeout=[], instance_ids={'instances': instance_ids, 'success': [], 'error': [], 'timeout': []})
        elif group is not None:
            if auto_increment:
                numbers = set()
                try:
                    name % 0
                except TypeError as e:
                    if e.message.startswith('not all'):
                        name = '%s%%d' % name
                    else:
                        module.fail_json(msg=e.message)
                pattern = re.sub('%\\d*[sd]', '(\\d+)', name)
                for server in cs.servers.list():
                    if server.status == 'DELETED':
                        continue
                    if server.metadata.get('group') == group:
                        servers.append(server)
                    match = re.search(pattern, server.name)
                    if match:
                        number = int(match.group(1))
                        numbers.add(number)
                number_range = xrange(count_offset, count_offset + count + len(numbers))
                available_numbers = list(set(number_range).difference(numbers))
                names = []
                numbers_to_use = available_numbers[:count]
                for number in numbers_to_use:
                    names.append(name % number)
            else:
                names = [name] * count
        else:
            search_opts = {'name': '^%s$' % name, 'flavor': flavor}
            servers = []
            for server in cs.servers.list(search_opts=search_opts):
                if server.status == 'DELETED':
                    continue
                if not rax_find_server_image(module, server, image, boot_volume):
                    continue
                if server.metadata != meta:
                    continue
                servers.append(server)
            if len(servers) >= count:
                instances = []
                for server in servers:
                    instances.append(rax_to_dict(server, 'server'))
                instance_ids = [i['id'] for i in instances]
                module.exit_json(changed=False, action=None, instances=instances, success=[], error=[], timeout=[], instance_ids={'instances': instance_ids, 'success': [], 'error': [], 'timeout': []})
            names = [name] * (count - len(servers))
        block_device_mapping_v2 = []
        if boot_from_volume:
            mapping = {'boot_index': '0', 'delete_on_termination': boot_volume_terminate, 'destination_type': 'volume'}
            if image:
                mapping.update({'uuid': image, 'source_type': 'image', 'volume_size': boot_volume_size})
                image = None
            elif boot_volume:
                volume = rax_find_volume(module, pyrax, boot_volume)
                mapping.update({'uuid': pyrax.utils.get_id(volume), 'source_type': 'volume'})
            block_device_mapping_v2.append(mapping)
        create(module, names=names, flavor=flavor, image=image, meta=meta, key_name=key_name, files=files, wait=wait, wait_timeout=wait_timeout, disk_config=disk_config, group=group, nics=nics, extra_create_args=extra_create_args, user_data=user_data, config_drive=config_drive, existing=servers, block_device_mapping_v2=block_device_mapping_v2)
    elif state == 'absent':
        if instance_ids is None:
            search_opts = {'name': '^%s$' % name, 'flavor': flavor}
            for server in cs.servers.list(search_opts=search_opts):
                if server.status == 'DELETED':
                    continue
                if not rax_find_server_image(module, server, image, boot_volume):
                    continue
                if meta != server.metadata:
                    continue
                servers.append(server)
            instance_ids = []
            for server in servers:
                if len(instance_ids) < count:
                    instance_ids.append(server.id)
                else:
                    break
        if not instance_ids:
            module.exit_json(changed=False, action=None, instances=[], success=[], error=[], timeout=[], instance_ids={'instances': [], 'success': [], 'error': [], 'timeout': []})
        delete(module, instance_ids=instance_ids, wait=wait, wait_timeout=wait_timeout)