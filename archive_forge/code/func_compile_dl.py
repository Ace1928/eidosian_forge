from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_dl(self, attdict, content, extra, program):
    atts = attdict.copy()
    atts = attdict.copy()
    bulletmaker = bulletMaker('dl', atts, self.context)
    contentcopy = list(content)
    bullet = ''
    while contentcopy:
        e = contentcopy[0]
        del contentcopy[0]
        if isinstance(e, str):
            if e.strip():
                raise ValueError("don't expect CDATA between list elements")
            elif not contentcopy:
                break
            else:
                continue
        elif isinstance(e, tuple):
            tagname, attdict1, content1, extra = e
            if tagname != 'dd' and tagname != 'dt':
                raise ValueError("don't expect %s here inside list, expect 'dd' or 'dt'" % repr(tagname))
            if tagname == 'dt':
                if bullet:
                    raise ValueError('dt will not be displayed unless followed by a dd: ' + repr(bullet))
                if content1:
                    self.compile_para(attdict1, content1, extra, program)
            elif tagname == 'dd':
                newatts = atts.copy()
                if attdict1:
                    newatts.update(attdict1)
                bulletmaker.makeBullet(newatts, bl=bullet)
                self.compile_para(newatts, content1, extra, program)
                bullet = ''
    if bullet:
        raise ValueError('dt will not be displayed unless followed by a dd' + repr(bullet))