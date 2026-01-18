from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def codex_list(module, skip_new=False):
    """ List valid grimoire collection. """
    params = module.params
    codex = {}
    cmd_scribe = '%s index' % SORCERY['scribe']
    rc, stdout, stderr = module.run_command(cmd_scribe)
    if rc != 0:
        module.fail_json(msg='unable to list grimoire collection, fix your Codex')
    rex = re.compile('^\\s*\\[\\d+\\] : (?P<grim>[\\w\\-+.]+) : [\\w\\-+./]+(?: : (?P<ver>[\\w\\-+.]+))?\\s*$')
    for line in stdout.splitlines()[4:-1]:
        match = rex.match(line)
        if match:
            codex[match.group('grim')] = match.group('ver')
    if params['repository'] and (not skip_new):
        codex = dict(((x, codex.get(x, NA)) for x in params['name']))
    if not codex:
        module.fail_json(msg='no grimoires to operate on; add at least one')
    return codex