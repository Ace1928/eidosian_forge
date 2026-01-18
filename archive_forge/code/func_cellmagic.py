import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@CoroutineInputTransformer.wrap
def cellmagic(end_on_blank_line: bool=False):
    """Captures & transforms cell magics.

    After a cell magic is started, this stores up any lines it gets until it is
    reset (sent None).
    """
    tpl = 'get_ipython().run_cell_magic(%r, %r, %r)'
    cellmagic_help_re = re.compile('%%\\w+\\?')
    line = ''
    while True:
        line = (yield line)
        while not line:
            line = (yield line)
        if not line.startswith(ESC_MAGIC2):
            while line is not None:
                line = (yield line)
            continue
        if cellmagic_help_re.match(line):
            continue
        first = line
        body = []
        line = (yield None)
        while line is not None and (line.strip() != '' or not end_on_blank_line):
            body.append(line)
            line = (yield None)
        magic_name, _, first = first.partition(' ')
        magic_name = magic_name.lstrip(ESC_MAGIC2)
        line = tpl % (magic_name, first, u'\n'.join(body))