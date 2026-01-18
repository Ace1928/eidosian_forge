from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def sensu_subscription(module, path, name, state='present', backup=False):
    changed = False
    reasons = []
    try:
        config = json.load(open(path))
    except IOError as e:
        if e.errno == 2:
            if state == 'absent':
                reasons.append("file did not exist and state is `absent'")
                return (changed, reasons)
            config = {}
        else:
            module.fail_json(msg=to_native(e), exception=traceback.format_exc())
    except ValueError:
        msg = '{path} contains invalid JSON'.format(path=path)
        module.fail_json(msg=msg)
    if 'client' not in config:
        if state == 'absent':
            reasons.append("`client' did not exist and state is `absent'")
            return (changed, reasons)
        config['client'] = {}
        changed = True
        reasons.append("`client' did not exist")
    if 'subscriptions' not in config['client']:
        if state == 'absent':
            reasons.append("`client.subscriptions' did not exist and state is `absent'")
            return (changed, reasons)
        config['client']['subscriptions'] = []
        changed = True
        reasons.append("`client.subscriptions' did not exist")
    if name not in config['client']['subscriptions']:
        if state == 'absent':
            reasons.append('channel subscription was absent')
            return (changed, reasons)
        config['client']['subscriptions'].append(name)
        changed = True
        reasons.append("channel subscription was absent and state is `present'")
    elif state == 'absent':
        config['client']['subscriptions'].remove(name)
        changed = True
        reasons.append("channel subscription was present and state is `absent'")
    if changed and (not module.check_mode):
        if backup:
            module.backup_local(path)
        try:
            open(path, 'w').write(json.dumps(config, indent=2) + '\n')
        except IOError as e:
            module.fail_json(msg='Failed to write to file %s: %s' % (path, to_native(e)), exception=traceback.format_exc())
    return (changed, reasons)