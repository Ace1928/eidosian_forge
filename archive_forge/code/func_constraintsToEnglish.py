import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def constraintsToEnglish(self, escapeLiteral=returnUnchanged, escapeNumber=returnUnchanged):
    constraints = []
    keyConstraints = self.key.constraintsToEnglish(escapeLiteral, escapeNumber)
    if keyConstraints:
        if self.value:
            constraints.append('key %s' % keyConstraints)
        else:
            constraints.append(keyConstraints)
    if self.value:
        valueConstraints = self.value.constraintsToEnglish(escapeLiteral, escapeNumber)
        if valueConstraints:
            constraints.append('value %s' % valueConstraints)
    return ', '.join(constraints)