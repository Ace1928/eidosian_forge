from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def change_device_attr(module, attributes, device, force):
    """ Change AIX device attribute. """
    attr_changed = []
    attr_not_changed = []
    attr_invalid = []
    chdev_cmd = module.get_bin_path('chdev', True)
    for attr in list(attributes.keys()):
        new_param = attributes[attr]
        current_param = _check_device_attr(module, device, attr)
        if current_param is None:
            attr_invalid.append(attr)
        elif current_param != new_param:
            if force:
                cmd = ['%s' % chdev_cmd, '-l', '%s' % device, '-a', '%s=%s' % (attr, attributes[attr]), '%s' % force]
            else:
                cmd = ['%s' % chdev_cmd, '-l', '%s' % device, '-a', '%s=%s' % (attr, attributes[attr])]
            if not module.check_mode:
                rc, chdev_out, err = module.run_command(cmd)
                if rc != 0:
                    module.exit_json(msg='Failed to run chdev.', rc=rc, err=err)
            attr_changed.append(attributes[attr])
        else:
            attr_not_changed.append(attributes[attr])
    if len(attr_changed) > 0:
        changed = True
        attr_changed_msg = 'Attributes changed: %s. ' % ','.join(attr_changed)
    else:
        changed = False
        attr_changed_msg = ''
    if len(attr_not_changed) > 0:
        attr_not_changed_msg = 'Attributes already set: %s. ' % ','.join(attr_not_changed)
    else:
        attr_not_changed_msg = ''
    if len(attr_invalid) > 0:
        attr_invalid_msg = 'Invalid attributes: %s ' % ', '.join(attr_invalid)
    else:
        attr_invalid_msg = ''
    msg = '%s%s%s' % (attr_changed_msg, attr_not_changed_msg, attr_invalid_msg)
    return (changed, msg)