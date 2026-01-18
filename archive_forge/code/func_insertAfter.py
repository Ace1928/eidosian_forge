from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def insertAfter(self, *args):
    if len(args) == 2:
        programName = self.DEFAULT_PROGRAM_NAME
        index = args[0]
        text = args[1]
    elif len(args) == 3:
        programName = args[0]
        index = args[1]
        text = args[2]
    else:
        raise TypeError('Invalid arguments')
    if isinstance(index, Token):
        index = index.index
    self.insertBefore(programName, index + 1, text)