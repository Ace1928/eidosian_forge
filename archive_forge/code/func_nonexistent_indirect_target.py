import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def nonexistent_indirect_target(self, target):
    if target['refname'] in self.document.nameids:
        self.indirect_target_error(target, 'which is a duplicate, and cannot be used as a unique reference')
    else:
        self.indirect_target_error(target, 'which does not exist')