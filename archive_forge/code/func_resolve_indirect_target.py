import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def resolve_indirect_target(self, target):
    refname = target.get('refname')
    if refname is None:
        reftarget_id = target['refid']
    else:
        reftarget_id = self.document.nameids.get(refname)
        if not reftarget_id:
            for resolver_function in self.document.transformer.unknown_reference_resolvers:
                if resolver_function(target):
                    break
            else:
                self.nonexistent_indirect_target(target)
            return
    reftarget = self.document.ids[reftarget_id]
    reftarget.note_referenced_by(id=reftarget_id)
    if isinstance(reftarget, nodes.target) and (not reftarget.resolved) and reftarget.hasattr('refname'):
        if hasattr(target, 'multiply_indirect'):
            self.circular_indirect_reference(target)
            return
        target.multiply_indirect = 1
        self.resolve_indirect_target(reftarget)
        del target.multiply_indirect
    if reftarget.hasattr('refuri'):
        target['refuri'] = reftarget['refuri']
        if 'refid' in target:
            del target['refid']
    elif reftarget.hasattr('refid'):
        target['refid'] = reftarget['refid']
        self.document.note_refid(target)
    elif reftarget['ids']:
        target['refid'] = reftarget_id
        self.document.note_refid(target)
    else:
        self.nonexistent_indirect_target(target)
        return
    if refname is not None:
        del target['refname']
    target.resolved = 1