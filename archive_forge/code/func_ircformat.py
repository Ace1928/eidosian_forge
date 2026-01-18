import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.util import get_choice_opt
def ircformat(color, text):
    if len(color) < 1:
        return text
    add = sub = ''
    if '_' in color:
        add += '\x1d'
        sub = '\x1d' + sub
        color = color.strip('_')
    if '*' in color:
        add += '\x02'
        sub = '\x02' + sub
        color = color.strip('*')
    if len(color) > 0:
        add += '\x03' + str(IRC_COLOR_MAP[color]).zfill(2)
        sub = '\x03' + sub
    return add + text + sub
    return '<' + add + '>' + text + '</' + sub + '>'