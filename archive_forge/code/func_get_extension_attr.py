import sys
import os
def get_extension_attr(self, extension, option_name, default=False):
    return getattr(self, option_name) or getattr(extension, option_name, default)