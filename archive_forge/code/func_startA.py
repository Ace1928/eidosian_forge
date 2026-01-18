from reportlab.lib import colors
@property
def startA(self):
    """Start position of Feature A."""
    try:
        return self.featureA.start
    except AttributeError:
        track, start, end = self.featureA
        return start