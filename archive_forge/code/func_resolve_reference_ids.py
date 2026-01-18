import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def resolve_reference_ids(self, target):
    """
        Given::

            <paragraph>
                <reference refname="direct internal">
                    direct internal
            <target id="id1" name="direct internal">

        The "refname" attribute is replaced by "refid" linking to the target's
        "id"::

            <paragraph>
                <reference refid="id1">
                    direct internal
            <target id="id1" name="direct internal">
        """
    for name in target['names']:
        refid = self.document.nameids.get(name)
        reflist = self.document.refnames.get(name, [])
        if reflist:
            target.note_referenced_by(name=name)
        for ref in reflist:
            if ref.resolved:
                continue
            if refid:
                del ref['refname']
                ref['refid'] = refid
            ref.resolved = 1