from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_a(self, attdict, content, extra, program):
    url = attdict['href']
    colorname = attdict.get('color', 'blue')
    Link = HotLink(url)
    program.append(('push',))
    program.append(('color', colorname))
    program.append(('lineOperation', Link))
    program.append(('lineOperation', UNDERLINE))
    for e in content:
        self.compileComponent(e, program)
    program.append(('endLineOperation', UNDERLINE))
    program.append(('endLineOperation', Link))
    program.append(('pop',))