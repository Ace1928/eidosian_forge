import logging
import re
def ltr(version, range_, loose):
    return outside(version, range_, '<', loose)