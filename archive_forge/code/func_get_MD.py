from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def get_MD(self, args, cmd=None, wd='branch'):
    md = self.run_send(args, cmd=cmd, wd=wd)[0]
    out = BytesIO(md.encode('utf-8'))
    return merge_directive.MergeDirective.from_lines(out)