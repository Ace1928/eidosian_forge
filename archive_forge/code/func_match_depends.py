from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def match_depends(module):
    """ Check for matching dependencies.

    This inspects spell's dependencies with the desired states and returns
    'False' if a recast is needed to match them. It also adds required lines
    to the system-wide depends file for proper recast procedure.

    """
    params = module.params
    spells = params['name']
    depends = {}
    depends_ok = True
    if len(spells) > 1 or not params['depends']:
        return depends_ok
    spell = spells[0]
    if module.check_mode:
        sorcery_depends_orig = os.path.join(SORCERY_STATE_DIR, 'depends')
        sorcery_depends = os.path.join(SORCERY_STATE_DIR, 'depends.check')
        try:
            shutil.copy2(sorcery_depends_orig, sorcery_depends)
        except IOError:
            module.fail_json(msg='failed to copy depends.check file')
    else:
        sorcery_depends = os.path.join(SORCERY_STATE_DIR, 'depends')
    rex = re.compile('^(?P<status>\\+?|\\-){1}(?P<depend>[a-z0-9]+[a-z0-9_\\-\\+\\.]*(\\([A-Z0-9_\\-\\+\\.]+\\))*)$')
    for d in params['depends'].split(','):
        match = rex.match(d)
        if not match:
            module.fail_json(msg="wrong depends line for spell '%s'" % spell)
        if not match.group('status') or match.group('status') == '+':
            status = 'on'
        else:
            status = 'off'
        depends[match.group('depend')] = status
    depends_list = [s.split('(')[0] for s in depends]
    cmd_gaze = '%s -q version %s' % (SORCERY['gaze'], ' '.join(depends_list))
    rc, stdout, stderr = module.run_command(cmd_gaze)
    if rc != 0:
        module.fail_json(msg="wrong dependencies for spell '%s'" % spell)
    fi = fileinput.input(sorcery_depends, inplace=True)
    try:
        try:
            for line in fi:
                if line.startswith(spell + ':'):
                    match = None
                    for d in depends:
                        d_offset = d.find('(')
                        if d_offset == -1:
                            d_p = ''
                        else:
                            d_p = re.escape(d[d_offset:])
                        rex = re.compile('%s:(?:%s|%s):(?P<lstatus>on|off):optional:' % (re.escape(spell), re.escape(d), d_p))
                        match = rex.match(line)
                        if match:
                            if match.group('lstatus') == depends[d]:
                                depends[d] = None
                                sys.stdout.write(line)
                            break
                    if not match:
                        sys.stdout.write(line)
                else:
                    sys.stdout.write(line)
        except IOError:
            module.fail_json(msg='I/O error on the depends file')
    finally:
        fi.close()
    depends_new = [v for v in depends if depends[v]]
    if depends_new:
        try:
            try:
                fl = open(sorcery_depends, 'a')
                for k in depends_new:
                    fl.write('%s:%s:%s:optional::\n' % (spell, k, depends[k]))
            except IOError:
                module.fail_json(msg='I/O error on the depends file')
        finally:
            fl.close()
        depends_ok = False
    if module.check_mode:
        try:
            os.remove(sorcery_depends)
        except IOError:
            module.fail_json(msg='failed to clean up depends.backup file')
    return depends_ok