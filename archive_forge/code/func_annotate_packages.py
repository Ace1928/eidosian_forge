from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def annotate_packages(module, run_pkgng, packages, annotations):
    annotate_c = 0
    if len(annotations) == 1:
        annotations = re.split('\\s*,\\s*', annotations[0])
    operation = {'+': annotation_add, '-': annotation_delete, ':': annotation_modify}
    for package in packages:
        for annotation_string in annotations:
            annotation = re.match('(?P<operation>[-+:])(?P<tag>[^=]+)(=(?P<value>.+))?', annotation_string)
            if annotation is None:
                module.fail_json(msg='failed to annotate %s, invalid annotate string: %s' % (package, annotation_string))
            annotation = annotation.groupdict()
            if operation[annotation['operation']](module, run_pkgng, package, annotation['tag'], annotation['value']):
                annotate_c += 1
    if annotate_c > 0:
        return (True, 'added %s annotations.' % annotate_c)
    return (False, 'changed no annotations')