import os
from .. import logging
def no_tvtk():
    """Checks if tvtk was found"""
    global _have_tvtk
    return not _have_tvtk