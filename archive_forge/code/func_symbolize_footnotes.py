import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def symbolize_footnotes(self):
    """Add symbols indexes to "[*]"-style footnotes and references."""
    labels = []
    for footnote in self.document.symbol_footnotes:
        reps, index = divmod(self.document.symbol_footnote_start, len(self.symbols))
        labeltext = self.symbols[index] * (reps + 1)
        labels.append(labeltext)
        footnote.insert(0, nodes.label('', labeltext))
        self.document.symbol_footnote_start += 1
        self.document.set_id(footnote)
    i = 0
    for ref in self.document.symbol_footnote_refs:
        try:
            ref += nodes.Text(labels[i])
        except IndexError:
            msg = self.document.reporter.error('Too many symbol footnote references: only %s corresponding footnotes available.' % len(labels), base_node=ref)
            msgid = self.document.set_id(msg)
            for ref in self.document.symbol_footnote_refs[i:]:
                if ref.resolved or ref.hasattr('refid'):
                    continue
                prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                prbid = self.document.set_id(prb)
                msg.add_backref(prbid)
                ref.replace_self(prb)
            break
        footnote = self.document.symbol_footnotes[i]
        assert len(footnote['ids']) == 1
        ref['refid'] = footnote['ids'][0]
        self.document.note_refid(ref)
        footnote.add_backref(ref['ids'][0])
        i += 1