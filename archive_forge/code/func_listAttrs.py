from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
def listAttrs(self, indent=''):
    print(indent + 'name =', self.name)
    print(indent + 'parent =', self.parent)
    keylist = list(self.__dict__.keys())
    keylist.sort()
    keylist.remove('name')
    keylist.remove('parent')
    for key in keylist:
        value = self.__dict__.get(key, None)
        print(indent + '%s = %s' % (key, value))