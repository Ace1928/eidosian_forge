from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def generate_systemd(module, module_params, name, version):
    result = {'changed': False, 'systemd': {}, 'diff': {}}
    sysconf = module_params['generate_systemd']
    rc, systemd, err = run_generate_systemd_command(module, module_params, name, version)
    if rc != 0:
        module.log('PODMAN-CONTAINER-DEBUG: Error generating systemd: %s' % err)
        if sysconf:
            module.fail_json(msg='Error generating systemd: %s' % err)
        return result
    else:
        try:
            data = json.loads(systemd)
            result['systemd'] = data
            if sysconf.get('path'):
                full_path = os.path.expanduser(sysconf['path'])
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                    result['changed'] = True
                if not os.path.isdir(full_path):
                    module.fail_json('Path %s is not a directory! Can not save systemd unit files there!' % full_path)
                for file_name, file_content in data.items():
                    file_name += '.service'
                    if not os.path.exists(os.path.join(full_path, file_name)):
                        result['changed'] = True
                        if result['diff'].get('before') is None:
                            result['diff'] = {'before': {}, 'after': {}}
                        result['diff']['before'].update({'systemd_{file_name}.service'.format(file_name=file_name): ''})
                        result['diff']['after'].update({'systemd_{file_name}.service'.format(file_name=file_name): file_content})
                    else:
                        diff_ = compare_systemd_file_content(os.path.join(full_path, file_name), file_content)
                        if diff_:
                            result['changed'] = True
                            if result['diff'].get('before') is None:
                                result['diff'] = {'before': {}, 'after': {}}
                            result['diff']['before'].update({'systemd_{file_name}.service'.format(file_name=file_name): '\n'.join(diff_[0])})
                            result['diff']['after'].update({'systemd_{file_name}.service'.format(file_name=file_name): '\n'.join(diff_[1])})
                    with open(os.path.join(full_path, file_name), 'w') as f:
                        f.write(file_content)
                diff_before = '\n'.join(['{j} - {k}'.format(j=j, k=k) for j, k in result['diff'].get('before', {}).items() if 'PIDFile' not in k]).strip()
                diff_after = '\n'.join(['{j} - {k}'.format(j=j, k=k) for j, k in result['diff'].get('after', {}).items() if 'PIDFile' not in k]).strip()
                if diff_before or diff_after:
                    result['diff']['before'] = diff_before + '\n'
                    result['diff']['after'] = diff_after + '\n'
                else:
                    result['diff'] = {}
            return result
        except Exception as e:
            module.log('PODMAN-CONTAINER-DEBUG: Error writing systemd: %s' % e)
            if sysconf:
                module.fail_json(msg='Error writing systemd: %s' % e)
            return result