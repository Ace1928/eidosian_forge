from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def getKindOfOps(self, rewrites, kind, before=None):
    if before is None:
        before = len(rewrites)
    elif before > len(rewrites):
        before = len(rewrites)
    for i, op in enumerate(rewrites[:before]):
        if op is None:
            continue
        if op.__class__ == kind:
            yield (i, op)