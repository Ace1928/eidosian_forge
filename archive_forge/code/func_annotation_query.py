from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def annotation_query(module, run_pkgng, package, tag):
    rc, out, err = run_pkgng('info', '-g', '-A', package)
    match = re.search('^\\s*(?P<tag>%s)\\s*:\\s*(?P<value>\\w+)' % tag, out, flags=re.MULTILINE)
    if match:
        return match.group('value')
    return False