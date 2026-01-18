from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compileComponent(self, parsedText, program):
    if isinstance(parsedText, str):
        if parsedText:
            stext = parsedText.strip()
            if not stext:
                program.append(' ')
            else:
                handleSpecialCharacters(self, parsedText, program)
    elif isinstance(parsedText, list):
        for e in parsedText:
            self.compileComponent(e, program)
    elif isinstance(parsedText, tuple):
        tagname, attdict, content, extra = parsedText
        if not attdict:
            attdict = {}
        compilername = 'compile_' + tagname
        compiler = getattr(self, compilername, None)
        if compiler is not None:
            compiler(attdict, content, extra, program)
        elif debug:
            L = ['<' + tagname]
            a = L.append
            if not attdict:
                attdict = {}
            for k, v in attdict.items():
                a(' %s=%s' % (k, v))
            if content:
                a('>')
                a(str(content))
                a('</%s>' % tagname)
            else:
                a('/>')
            t = ''.join(L)
            handleSpecialCharacters(self, t, program)
        else:
            raise ValueError("don't know how to handle tag " + repr(tagname))