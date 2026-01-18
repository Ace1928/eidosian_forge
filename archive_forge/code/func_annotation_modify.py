from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def annotation_modify(module, run_pkgng, package, tag, value):
    _value = annotation_query(module, run_pkgng, package, tag)
    if not _value:
        module.fail_json(msg='could not change annotation to %s: tag %s does not exist' % (package, tag))
    elif _value == value:
        return False
    else:
        if not module.check_mode:
            rc, out, err = run_pkgng('annotate', '-y', '-M', package, tag, data=value, binary_data=True)
            if rc != 0 and re.search('^%s-[^:]+: Modified annotation tagged: %s' % (package, tag), out, flags=re.MULTILINE) is None:
                module.fail_json(msg='failed to annotate %s, could not change annotation %s to %s: %s' % (package, tag, value, out), stderr=err)
        return True