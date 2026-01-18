import re, time, datetime
from .utils import isStr
def localeFormat(self):
    """override this method to use your preferred locale format"""
    return self.formatUS()