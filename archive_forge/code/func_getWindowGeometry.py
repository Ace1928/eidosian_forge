import Quartz
import pygetwindow
def getWindowGeometry(title):
    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListExcludeDesktopElements | Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    for win in windows:
        if title in '%s %s' % (win[Quartz.kCGWindowOwnerName], win.get(Quartz.kCGWindowName, '')):
            w = win['kCGWindowBounds']
            return (w['X'], w['Y'], w['Width'], w['Height'])