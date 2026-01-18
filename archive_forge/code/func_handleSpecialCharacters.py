from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def handleSpecialCharacters(engine, text, program=None):
    from reportlab.platypus.paraparser import greeks
    from string import whitespace
    standard = {'lt': '<', 'gt': '>', 'amp': '&'}
    if text[0:1] in whitespace:
        program.append(' ')
    if 0 and '&' not in text:
        result = []
        for x in text.split():
            result.append(x + ' ')
        if result:
            last = result[-1]
            if text[-1:] not in whitespace:
                result[-1] = last.strip()
        program.extend(result)
        return program
    if program is None:
        program = []
    amptext = text.split('&')
    first = 1
    lastfrag = amptext[-1]
    for fragment in amptext:
        if not first:
            semi = fragment.find(';')
            if semi > 0:
                name = fragment[:semi]
                if name[0] == '#':
                    try:
                        if name[1] == 'x':
                            n = int(name[2:], 16)
                        else:
                            n = int(name[1:])
                    except ValueError:
                        n = -1
                    if n >= 0:
                        fragment = chr(n) + fragment[semi + 1:]
                    else:
                        fragment = '&' + fragment
                elif name in standard:
                    s = standard[name]
                    if isinstance(fragment, bytes):
                        s = s.decode('utf8')
                    fragment = s + fragment[semi + 1:]
                elif name in greeks:
                    s = greeks[name]
                    if isinstance(fragment, bytes):
                        s = s.decode('utf8')
                    fragment = s + fragment[semi + 1:]
                else:
                    fragment = '&' + fragment
            else:
                fragment = '&' + fragment
        sfragment = fragment.split()
        for w in sfragment[:-1]:
            program.append(w + ' ')
        if sfragment and fragment:
            if fragment[-1] in whitespace:
                program.append(sfragment[-1] + ' ')
            else:
                last = sfragment[-1].strip()
                if last:
                    program.append(last)
        first = 0
    return program