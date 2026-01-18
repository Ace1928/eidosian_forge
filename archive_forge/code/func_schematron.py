import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
@property
def schematron(self):
    """ISO-schematron schema document (None if object has been initialized
        with store_schematron=False).
        """
    return self._schematron