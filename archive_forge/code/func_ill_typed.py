import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
@property
def ill_typed(self) -> Optional[bool]:
    """
        For `recognized datatype IRIs
        <https://www.w3.org/TR/rdf11-concepts/#dfn-recognized-datatype-iris>`_,
        this value will be `True` if the literal is ill formed, otherwise it
        will be `False`. `Literal.value` (i.e. the `literal value <https://www.w3.org/TR/rdf11-concepts/#dfn-literal-value>`_) should always be defined if this property is `False`, but should not be considered reliable if this property is `True`.

        If the literal's datatype is `None` or not in the set of `recognized datatype IRIs
        <https://www.w3.org/TR/rdf11-concepts/#dfn-recognized-datatype-iris>`_ this value will be `None`.
        """
    return self._ill_typed