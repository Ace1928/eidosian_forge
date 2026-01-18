from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def get_lint(self):
    """Return lint records in sorted order"""
    records = [record for _, _, _, record in sorted(((record.location, record.spec.idstr, idx, record) for idx, record in enumerate(self._lint)))]
    out = []
    events = list(self._suppression_events)
    active_suppressions = set()
    for record in records:
        if record.location:
            while events and record.location[0] >= events[0].lineno:
                event = events.pop(0)
                if event.mode == 'add':
                    for idstr in event.suppressions:
                        active_suppressions.add(idstr)
                elif event.mode == 'remove':
                    for idstr in event.suppressions:
                        active_suppressions.discard(idstr)
                else:
                    raise ValueError('Illegal suppression event {}'.format(event.mode))
        if record.spec.idstr in active_suppressions:
            continue
        out.append(record)
    return out