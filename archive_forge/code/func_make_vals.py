import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def make_vals(val, klass, klass_inst=None, prop=None, part=False, base64encode=False):
    """
    Creates a class instance with a specified value, the specified
    class instance may be a value on a property in a defined class instance.

    :param val: The value
    :param klass: The value class
    :param klass_inst: The class instance which has a property on which
        what this function returns is a value.
    :param prop: The property which the value should be assigned to.
    :param part: If the value is one of a possible list of values it should be
        handled slightly different compared to if it isn't.
    :return: Value class instance
    """
    cinst = None
    if isinstance(val, dict):
        cinst = klass().loadd(val, base64encode=base64encode)
    else:
        try:
            cinst = klass().set_text(val)
        except ValueError:
            if not part:
                cis = [make_vals(sval, klass, klass_inst, prop, True, base64encode) for sval in val]
                setattr(klass_inst, prop, cis)
            else:
                raise
    if part:
        return cinst
    elif cinst:
        cis = [cinst]
        setattr(klass_inst, prop, cis)