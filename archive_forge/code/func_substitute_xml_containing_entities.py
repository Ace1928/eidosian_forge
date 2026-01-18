from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def substitute_xml_containing_entities(cls, value, make_quoted_attribute=False):
    """Substitute XML entities for special XML characters.

        :param value: A string to be substituted. The less-than sign will
          become &lt;, the greater-than sign will become &gt;, and any
          ampersands that are not part of an entity defition will
          become &amp;.

        :param make_quoted_attribute: If True, then the string will be
         quoted, as befits an attribute value.
        """
    value = cls.BARE_AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)
    if make_quoted_attribute:
        value = cls.quoted_attribute_value(value)
    return value