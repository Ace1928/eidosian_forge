from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def cjkwrap(text, width, encoding='utf8'):
    return reduce(lambda line, word, width=width: '%s%s%s' % (line, [' ', '\n', ''][len(line) - line.rfind('\n') - 1 + len(word.split('\n', 1)[0]) >= width or (line[-1:] == '\x00' and 2)], word), rx.sub('\\1\\0 ', str(text, encoding)).split(' ')).replace('\x00', '').encode(encoding)