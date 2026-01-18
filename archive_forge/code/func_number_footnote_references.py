import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def number_footnote_references(self, startnum):
    """Assign numbers to autonumbered footnote references."""
    i = 0
    for ref in self.document.autofootnote_refs:
        if ref.resolved or ref.hasattr('refid'):
            continue
        try:
            label = self.autofootnote_labels[i]
        except IndexError:
            msg = self.document.reporter.error('Too many autonumbered footnote references: only %s corresponding footnotes available.' % len(self.autofootnote_labels), base_node=ref)
            msgid = self.document.set_id(msg)
            for ref in self.document.autofootnote_refs[i:]:
                if ref.resolved or ref.hasattr('refname'):
                    continue
                prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                prbid = self.document.set_id(prb)
                msg.add_backref(prbid)
                ref.replace_self(prb)
            break
        ref += nodes.Text(label)
        id = self.document.nameids[label]
        footnote = self.document.ids[id]
        ref['refid'] = id
        self.document.note_refid(ref)
        assert len(ref['ids']) == 1
        footnote.add_backref(ref['ids'][0])
        ref.resolved = 1
        i += 1