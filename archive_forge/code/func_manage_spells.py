from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def manage_spells(module):
    """ Cast or dispel spells.

    This manages the whole system ('*'), list or a single spell. Command 'cast'
    is used to install or rebuild spells, while 'dispel' takes care of theirs
    removal from the system.

    """
    params = module.params
    spells = params['name']
    sorcery_queue = os.path.join(SORCERY_LOG_DIR, 'queue/install')
    if spells == '*':
        if params['state'] == 'latest':
            try:
                os.rename(sorcery_queue, sorcery_queue + '.backup')
            except IOError:
                module.fail_json(msg='failed to backup the update queue')
            module.run_command_environ_update.update(dict(SILENT='1'))
            cmd_sorcery = '%s queue' % SORCERY['sorcery']
            rc, stdout, stderr = module.run_command(cmd_sorcery)
            if rc != 0:
                module.fail_json(msg='failed to generate the update queue')
            try:
                queue_size = os.stat(sorcery_queue).st_size
            except Exception:
                module.fail_json(msg='failed to read the update queue')
            if queue_size != 0:
                if module.check_mode:
                    try:
                        os.rename(sorcery_queue + '.backup', sorcery_queue)
                    except IOError:
                        module.fail_json(msg='failed to restore the update queue')
                    return (True, 'would have updated the system')
                cmd_cast = '%s --queue' % SORCERY['cast']
                rc, stdout, stderr = module.run_command(cmd_cast)
                if rc != 0:
                    module.fail_json(msg='failed to update the system')
                return (True, 'successfully updated the system')
            else:
                return (False, 'the system is already up to date')
        elif params['state'] == 'rebuild':
            if module.check_mode:
                return (True, 'would have rebuilt the system')
            cmd_sorcery = '%s rebuild' % SORCERY['sorcery']
            rc, stdout, stderr = module.run_command(cmd_sorcery)
            if rc != 0:
                module.fail_json(msg='failed to rebuild the system: ' + stdout)
            return (True, 'successfully rebuilt the system')
        else:
            module.fail_json(msg="unsupported operation on '*' name value")
    elif params['state'] in ('present', 'latest', 'rebuild', 'absent'):
        cmd_gaze = '%s -q version %s' % (SORCERY['gaze'], ' '.join(spells))
        rc, stdout, stderr = module.run_command(cmd_gaze)
        if rc != 0:
            module.fail_json(msg='failed to locate spell(s) in the list (%s)' % ', '.join(spells))
        cast_queue = []
        dispel_queue = []
        rex = re.compile('[^|]+\\|[^|]+\\|(?P<spell>[^|]+)\\|(?P<grim_ver>[^|]+)\\|(?P<inst_ver>[^$]+)')
        for line in stdout.splitlines()[2:-1]:
            match = rex.match(line)
            cast = False
            if params['state'] == 'present':
                if match.group('inst_ver') == '-':
                    match_depends(module)
                    cast = True
                elif not match_depends(module):
                    cast = True
            elif params['state'] == 'latest':
                if match.group('grim_ver') != match.group('inst_ver'):
                    match_depends(module)
                    cast = True
                elif not match_depends(module):
                    cast = True
            elif params['state'] == 'rebuild':
                cast = True
            elif match.group('inst_ver') != '-':
                dispel_queue.append(match.group('spell'))
            if cast:
                cast_queue.append(match.group('spell'))
        if cast_queue:
            if module.check_mode:
                return (True, 'would have cast spell(s)')
            cmd_cast = '%s -c %s' % (SORCERY['cast'], ' '.join(cast_queue))
            rc, stdout, stderr = module.run_command(cmd_cast)
            if rc != 0:
                module.fail_json(msg='failed to cast spell(s): ' + stdout)
            return (True, 'successfully cast spell(s)')
        elif params['state'] != 'absent':
            return (False, 'spell(s) are already cast')
        if dispel_queue:
            if module.check_mode:
                return (True, 'would have dispelled spell(s)')
            cmd_dispel = '%s %s' % (SORCERY['dispel'], ' '.join(dispel_queue))
            rc, stdout, stderr = module.run_command(cmd_dispel)
            if rc != 0:
                module.fail_json(msg='failed to dispel spell(s): ' + stdout)
            return (True, 'successfully dispelled spell(s)')
        else:
            return (False, 'spell(s) are already dispelled')