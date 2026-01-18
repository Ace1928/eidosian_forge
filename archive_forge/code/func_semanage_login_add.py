from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_login_add(module, login, seuser, do_reload, serange='s0', sestore=''):
    """ Add linux user to SELinux user mapping

    :type module: AnsibleModule
    :param module: Ansible module

    :type login: str
    :param login: a Linux User or a Linux group if it begins with %

    :type seuser: str
    :param proto: An SELinux user ('__default__', 'unconfined_u', 'staff_u', ...), see 'semanage login -l'

    :type serange: str
    :param serange: SELinux MLS/MCS range (defaults to 's0')

    :type do_reload: bool
    :param do_reload: Whether to reload SELinux policy after commit

    :type sestore: str
    :param sestore: SELinux store

    :rtype: bool
    :return: True if the policy was changed, otherwise False
    """
    try:
        selogin = seobject.loginRecords(sestore)
        selogin.set_reload(do_reload)
        change = False
        all_logins = selogin.get_all()
        if login not in all_logins.keys():
            change = True
            if not module.check_mode:
                selogin.add(login, seuser, serange)
        elif all_logins[login][0] != seuser or all_logins[login][1] != serange:
            change = True
            if not module.check_mode:
                selogin.modify(login, seuser, serange)
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        module.fail_json(msg='%s: %s\n' % (e.__class__.__name__, to_native(e)), exception=traceback.format_exc())
    return change