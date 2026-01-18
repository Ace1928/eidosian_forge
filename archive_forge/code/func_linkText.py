from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def linkText(text: str | None) -> None:
    if text:
        if result:
            if result[-1][0].tail:
                result[-1][0].tail += text
            else:
                result[-1][0].tail = text
        elif not isText:
            if parent.tail:
                parent.tail += text
            else:
                parent.tail = text
        elif parent.text:
            parent.text += text
        else:
            parent.text = text