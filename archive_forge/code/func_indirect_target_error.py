import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def indirect_target_error(self, target, explanation):
    naming = ''
    reflist = []
    if target['names']:
        naming = '"%s" ' % target['names'][0]
    for name in target['names']:
        reflist.extend(self.document.refnames.get(name, []))
    for id in target['ids']:
        reflist.extend(self.document.refids.get(id, []))
    if target['ids']:
        naming += '(id="%s")' % target['ids'][0]
    msg = self.document.reporter.error('Indirect hyperlink target %s refers to target "%s", %s.' % (naming, target['refname'], explanation), base_node=target)
    msgid = self.document.set_id(msg)
    for ref in utils.uniq(reflist):
        prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
        prbid = self.document.set_id(prb)
        msg.add_backref(prbid)
        ref.replace_self(prb)
    target.resolved = 1