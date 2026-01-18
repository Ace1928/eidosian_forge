from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def manage_grimoires(module):
    """ Add or remove grimoires. """
    params = module.params
    grimoires = params['name']
    url = params['repository']
    codex = codex_list(module, True)
    if url == '*':
        if params['state'] in ('present', 'latest', 'absent'):
            if params['state'] == 'absent':
                action = 'remove'
                todo = set(grimoires) & set(codex)
            else:
                action = 'add'
                todo = set(grimoires) - set(codex)
            if not todo:
                return (False, 'all grimoire(s) are already %sed' % action[:5])
            if module.check_mode:
                return (True, 'would have %sed grimoire(s)' % action[:5])
            cmd_scribe = '%s %s %s' % (SORCERY['scribe'], action, ' '.join(todo))
            rc, stdout, stderr = module.run_command(cmd_scribe)
            if rc != 0:
                module.fail_json(msg='failed to %s one or more grimoire(s): %s' % (action, stdout))
            return (True, 'successfully %sed one or more grimoire(s)' % action[:5])
        else:
            module.fail_json(msg="unsupported operation on '*' repository value")
    elif params['state'] in ('present', 'latest'):
        if len(grimoires) > 1:
            module.fail_json(msg='using multiple items with repository is invalid')
        grimoire = grimoires[0]
        if grimoire in codex:
            return (False, 'grimoire %s already exists' % grimoire)
        if module.check_mode:
            return (True, 'would have added grimoire %s from %s' % (grimoire, url))
        cmd_scribe = '%s add %s from %s' % (SORCERY['scribe'], grimoire, url)
        rc, stdout, stderr = module.run_command(cmd_scribe)
        if rc != 0:
            module.fail_json(msg='failed to add grimoire %s from %s: %s' % (grimoire, url, stdout))
        return (True, 'successfully added grimoire %s from %s' % (grimoire, url))
    else:
        module.fail_json(msg='unsupported operation on repository value')