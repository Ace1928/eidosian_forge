from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setLocale(self, locale):
    """Allow an application to set the locale for errors and warnings.

        SAX parsers are not required to provide localization for errors
        and warnings; if they cannot support the requested locale,
        however, they must raise a SAX exception. Applications may
        request a locale change in the middle of a parse."""
    raise SAXNotSupportedException('Locale support not implemented')