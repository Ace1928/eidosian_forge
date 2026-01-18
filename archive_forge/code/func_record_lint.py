from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def record_lint(self, idstr, *args, **kwargs):
    if idstr in self.config.lint.disabled_codes:
        self._supressed_count[idstr] += 1
        return
    if idstr in self._suppressions:
        self._supressed_count[idstr] += 1
        return
    spec = self.global_ctx.lintdb[idstr]
    location = kwargs.pop('location', ())
    msg = spec.msgfmt.format(*args, **kwargs)
    record = LintRecord(spec, location, msg)
    self._lint.append(record)