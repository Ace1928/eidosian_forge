import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def format_as_single_line(self, prefix=None, divider=' | ', enabled_item_attrs=None, disabled_item_attrs=None):
    """Format the menu as a single-line RichTextLines object.

    Args:
      prefix: (str) String added to the beginning of the line.
      divider: (str) The dividing string between the menu items.
      enabled_item_attrs: (list or str) Attributes applied to each enabled
        menu item, e.g., ["bold", "underline"].
      disabled_item_attrs: (list or str) Attributes applied to each
        disabled menu item, e.g., ["red"].

    Returns:
      (RichTextLines) A single-line output representing the menu, with
        font_attr_segs marking the individual menu items.
    """
    if enabled_item_attrs is not None and (not isinstance(enabled_item_attrs, list)):
        enabled_item_attrs = [enabled_item_attrs]
    if disabled_item_attrs is not None and (not isinstance(disabled_item_attrs, list)):
        disabled_item_attrs = [disabled_item_attrs]
    menu_line = prefix if prefix is not None else ''
    attr_segs = []
    for item in self._items:
        menu_line += item.caption
        item_name_begin = len(menu_line) - len(item.caption)
        if item.is_enabled():
            final_attrs = [item]
            if enabled_item_attrs:
                final_attrs.extend(enabled_item_attrs)
            attr_segs.append((item_name_begin, len(menu_line), final_attrs))
        elif disabled_item_attrs:
            attr_segs.append((item_name_begin, len(menu_line), disabled_item_attrs))
        menu_line += divider
    return RichTextLines(menu_line, font_attr_segs={0: attr_segs})