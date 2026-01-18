from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def assertSendSucceeds(self, args, revs=None, with_warning=False):
    if with_warning:
        err_re = self._default_errors
    else:
        err_re = []
    if revs is None:
        revs = self._default_sent_revs or [self.local]
    out, err = self.run_send(args, err_re=err_re)
    if len(revs) == 1:
        bundling_revs = 'Bundling %d revision.\n' % len(revs)
    else:
        bundling_revs = 'Bundling %d revisions.\n' % len(revs)
    if with_warning:
        self.assertContainsRe(err, self._default_additional_warning)
        self.assertEndsWith(err, bundling_revs)
    else:
        self.assertEqual(bundling_revs, err)
    md = merge_directive.MergeDirective.from_lines(BytesIO(out.encode('utf-8')))
    self.assertEqual(self.parent, md.base_revision_id)
    br = serializer.read_bundle(BytesIO(md.get_raw_bundle()))
    self.assertEqual(set(revs), {r.revision_id for r in br.revisions})